"""HDDM (Hoeffding's inequality drift detection method) module."""

import copy
from typing import List, Optional, Tuple, Union

import numpy as np  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.statistical_process_control.base import (
    BaseSPCConfig,
    BaseSPC,
)
from frouros.utils.stats import EWMA, Mean


class BaseHDDMConfig(BaseSPCConfig):
    """HDDM (Hoeffding's drift detection method) [frias2014online]_ configuration.

    :References:

    .. [frias2014online] Frias-Blanco, Isvani, et al.
        "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
        IEEE Transactions on Knowledge and Data Engineering 27.3 (2014): 810-823.
    """

    def __init__(
        self,
        alpha_d: float = 0.001,
        alpha_w: float = 0.005,
        two_sided_test: bool = False,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha_d: significance value for drift
        :type alpha_d: float
        :param alpha_w: significance value for warning
        :type alpha_w: float
        :param two_sided_test: flag that indicates if a two-sided test is performed
        :param two_sided_test: bool
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.alpha_d = alpha_d
        self.alpha_w = alpha_w
        self.two_sided_test = two_sided_test

    @property
    def alpha_d(self) -> float:
        """Significance level d property.

        :return: significance level d
        :rtype: float
        """
        return self._alpha_d

    @alpha_d.setter
    def alpha_d(self, value: float) -> None:
        """Significance level d setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 < value <= 1.0:
            raise ValueError("alpha_d must be in the range (0, 1].")
        self._alpha_d = value

    @property
    def alpha_w(self) -> float:
        """Significance level w property.

        :return: significance level w
        :rtype: float
        """
        return self._alpha_w

    @alpha_w.setter
    def alpha_w(self, value: float) -> None:
        """Significance level w setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 < value <= 1.0:
            raise ValueError("alpha_w must be in the range (0, 1].")
        if value <= self.alpha_d:
            raise ValueError("alpha_w must be greater than alpha_d.")
        self._alpha_w = value

    @property
    def two_sided_test(self) -> bool:
        """Two-sided test flag property.

        :return: two-sided test flag
        :rtype: float
        """
        return self._two_sided_test

    @two_sided_test.setter
    def two_sided_test(self, value: bool) -> None:
        """Two-sided test flag setter.

        :param value: value to be set
        :type value: bool
        :raises ValueError: Value error exception
        """
        if not isinstance(value, bool):
            raise ValueError("two_sided_test must be of type bool.")
        self._two_sided_test = value


class HDDMAConfig(BaseHDDMConfig):
    """HDDM-A (Hoeffding's drift detection method A-Test) [frias2014online]_ configuration.

    :References:

    .. [frias2014online] Frias-Blanco, Isvani, et al.
        "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
        IEEE Transactions on Knowledge and Data Engineering 27.3 (2014):
        810-823.
    """


class HDDMWConfig(BaseHDDMConfig):
    """HDDM-W (Hoeffding's drift detection method W-Test) [frias2014online]_ configuration.

    :References:

    .. [frias2014online] Frias-Blanco, Isvani, et al.
        "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
        IEEE Transactions on Knowledge and Data Engineering 27.3 (2014):
        810-823.
    """

    def __init__(
        self,
        alpha_d: float = 0.001,
        alpha_w: float = 0.005,
        two_sided_test: bool = False,
        lambda_: float = 0.05,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha_d: significance value for drift
        :type alpha_d: float
        :param alpha_w: significance value for warning
        :type alpha_w: float
        :param two_sided_test: flag that indicates if a two-sided test is performed
        :param two_sided_test: bool
        :param lambda_: weight given to recent data compared to older data
        :type lambda_: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(
            alpha_d=alpha_d,
            alpha_w=alpha_w,
            two_sided_test=two_sided_test,
            min_num_instances=min_num_instances,
        )
        self.lambda_ = lambda_

    @property
    def lambda_(self) -> float:
        """Weight recent data property.

        :return: weight given to recent data
        :rtype: float
        """
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value: float) -> None:
        """Weight recent data setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("lambda_ must be in the range [0, 1].")
        self._lambda_ = value


class HoeffdingOneSidedTest:
    """Hoeffding one-sided test class."""

    def __init__(self, alpha_d: float, alpha_w: float) -> None:
        """Init method."""
        self.alpha_d = alpha_d
        self.alpha_w = alpha_w
        self.x = Mean()
        self.z = Mean()

    def _check_mean_increase(self, m: int, alpha: float) -> bool:
        threshold = np.sqrt(
            m / (2 * self.x.num_values * self.z.num_values) * np.log(1 / alpha)
        )
        return self.z.mean - self.x.mean >= threshold

    def check_cases(self) -> Tuple[bool, bool]:
        """Check drift and warning cases.

        :return: drift and warning flags
        :rtype: Tuple[bool, bool]
        """
        m = self.z.num_values - self.x.num_values
        if self._check_mean_increase(m=m, alpha=self.alpha_d):
            drift = True
            warning = False
        elif self._check_mean_increase(m=m, alpha=self.alpha_w):
            drift = False
            warning = True
        else:
            drift = False
            warning = False

        return drift, warning

    def hoeffding_error_bound(self, num_values: int) -> float:
        """Calculate Hoeffding's error.

        :param num_values: number of values
        :type num_values: int
        :return: Hoeffding's error value
        :rtype: float
        """
        return np.sqrt(np.log(1 / self.alpha_d) / (2 * num_values))

    def reset(self) -> None:
        """Reset x and z incremental variables."""
        self.x = Mean()
        self.z = Mean()

    def set_initial_cut_mean(self) -> None:
        """Copy value of z to x if x has no values."""
        if self.x.num_values == 0:
            self.x = copy.deepcopy(self.z)

    def update_cut_point(self, epsilon_z: float) -> None:
        """Update cut point using Hoeffding's error.

        :param epsilon_z: Hoeffding's error of z
        :type epsilon_z: float
        """
        epsilon_x = self.hoeffding_error_bound(num_values=self.x.num_values)
        if self.z.mean + epsilon_z <= self.x.mean + epsilon_x:
            self.x = copy.deepcopy(self.z)


class HoeffdingTwoSidedTest(HoeffdingOneSidedTest):
    """Hoeffding two-sided test class."""

    def __init__(self, alpha_d: float, alpha_w: float) -> None:
        """Init method.

        :param alpha_d: significance value for drift
        :type alpha_d: float
        :param alpha_w: significance value for warning
        :type alpha_w: float
        """
        super().__init__(alpha_d=alpha_d, alpha_w=alpha_w)
        self.y = Mean()

    def reset(self) -> None:
        """Reset x, z and y incremental variables."""
        super().reset()
        self.y = Mean()

    def _check_mean_decrease(self, m: int, alpha: float) -> bool:
        threshold = np.sqrt(
            m / (2 * self.y.num_values * self.z.num_values) * np.log(1 / alpha)
        )
        return self.y.mean - self.z.mean >= threshold

    def check_cases(self) -> Tuple[bool, bool]:
        """Check drift and warning cases.

        :return: drift and warning flags
        :rtype: Tuple[bool, bool]
        """
        drift_increase, warning_increase = super().check_cases()
        m = self.z.num_values - self.y.num_values
        if self._check_mean_decrease(m=m, alpha=self.alpha_d):
            drift_decrease = True
            warning_decrease = False
        elif self._check_mean_decrease(m=m, alpha=self.alpha_w):
            drift_decrease = False
            warning_decrease = True
        else:
            drift_decrease = False
            warning_decrease = False
        return any([drift_increase, drift_decrease]), any(
            [warning_increase, warning_decrease]
        )

    def set_initial_cut_mean(self) -> None:
        """Copy value of z to x and/or y if x and/or y has no values."""
        super().set_initial_cut_mean()
        if self.y.num_values == 0:
            self.y = copy.deepcopy(self.z)

    def update_cut_point(self, epsilon_z: float) -> None:
        """Update cut point using Hoeffding's error.

        :param epsilon_z: Hoeffding's error of z
        :type epsilon_z: float
        """
        super().update_cut_point(epsilon_z=epsilon_z)
        epsilon_y = self.hoeffding_error_bound(num_values=self.y.num_values)
        if self.y.mean - epsilon_y <= self.z.mean - epsilon_z:
            self.y = copy.deepcopy(self.z)


class HDDMA(BaseSPC):
    """HDDM-A (Hoeffding's drift detection method with A-Test) [frias2014online]_ detector.

    :References:

    .. [frias2014online] Frias-Blanco, Isvani, et al.
        "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
        IEEE Transactions on Knowledge and Data Engineering 27.3 (2014):
        810-823.
    """

    config_type = HDDMAConfig  # type: ignore

    def __init__(
        self,
        config: Optional[HDDMAConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[HDDMAConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        """
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "test_type": (
                HoeffdingTwoSidedTest(
                    alpha_d=self.config.alpha_d,  # type: ignore
                    alpha_w=self.config.alpha_w,  # type: ignore
                )
                if self.config.two_sided_test  # type: ignore
                else HoeffdingOneSidedTest(
                    alpha_d=self.config.alpha_d,  # type: ignore
                    alpha_w=self.config.alpha_w,  # type: ignore
                )
            ),
            **self.additional_vars,  # type: ignore
        }
        self._set_additional_vars_callback()

    @property
    def test_type(self) -> HoeffdingOneSidedTest:
        """Test type property.

        :return: test type value
        :rtype: HoeffdingOneSidedTest
        """
        return self._additional_vars["test_type"]

    def _update(self, value: Union[int, float], **kwargs) -> None:
        self.num_instances += 1

        self.test_type.z.update(value=value)
        self.test_type.set_initial_cut_mean()

        epsilon_z = self.test_type.hoeffding_error_bound(
            num_values=self.test_type.z.num_values
        )
        self.test_type.update_cut_point(epsilon_z=epsilon_z)

        if self.num_instances >= self.config.min_num_instances:
            drift_flag, warning_flag = (
                self.test_type.check_cases()
                if self.test_type.z.num_values != self.test_type.x.num_values
                else (False, False)
            )
            if drift_flag:
                # Out-of-Control
                self.drift = True
                self.warning = False
                self.test_type.reset()
            else:
                if warning_flag:
                    # Warning
                    self.warning = True
                else:
                    # In-Control
                    self.warning = False
                self.drift = False
        else:
            self.drift, self.warning = False, False


class SampleInfo:
    """Sample information class."""

    def __init__(self, lambda_: float) -> None:
        """Init method.

        :param lambda_: weight given to recent data compared to older data
        :type lambda_: float
        """
        self.ewma = EWMA(alpha=lambda_)
        self.independent_bound_condition = 1.0
        self.lambda_squared = lambda_ * lambda_
        self.one_minus_lambda_squared = (
            self.ewma.one_minus_alpha * self.ewma.one_minus_alpha
        )

    @property
    def ewma(self) -> EWMA:
        """EWMA (Exponential Moving Average) property.

        :return: EWMA value
        :rtype: EWMA
        """
        return self._ewma

    @ewma.setter
    def ewma(self, value: EWMA) -> None:
        """EWMA (Exponential Moving Average)  setter.

        :param value: value to be set
        :type value: EWMA
        :raises TypeError: Type error exception
        """
        if not isinstance(value, EWMA):
            raise TypeError("ewma must be of type EWMA.")
        self._ewma = value

    def update(self, value: float) -> None:
        """Update statistics.

        :param value: value to use in update
        :type value: float
        """
        self.ewma.update(value=value)
        self.independent_bound_condition = (
            self.lambda_squared
            + self.one_minus_lambda_squared * self.independent_bound_condition
        )


class McDiarmidOneSidedTest:
    """McDiarmid one-sided test class."""

    def __init__(self, alpha_d: float, alpha_w: float, lambda_: float) -> None:
        """Init method.

        :param alpha_d: significance value for drift
        :type alpha_d: float
        :param alpha_w: significance value for warning
        :type alpha_w: float
        :param lambda_: weight given to recent data compared to older data
        :type lambda_: float
        """
        self.alpha_d = alpha_d
        self.alpha_w = alpha_w
        self.lambda_ = lambda_
        self.total = SampleInfo(lambda_=self.lambda_)
        self.sample_increase_1 = SampleInfo(lambda_=self.lambda_)
        self.sample_increase_2 = SampleInfo(lambda_=self.lambda_)
        self.increase_cut_point = float("inf")

    def _check_threshold(
        self, sample_1: SampleInfo, sample_2: SampleInfo, alpha: float
    ) -> bool:
        independent_bound_condition_sum = (
            sample_1.independent_bound_condition + sample_2.independent_bound_condition
        )
        threshold = self._mcdiarmid_error_bound(
            independent_bound_condition=independent_bound_condition_sum,
            alpha=alpha,
        )
        return sample_2.ewma.mean - sample_1.ewma.mean > threshold

    def _check_mean_increase(self, alpha: float) -> bool:
        return self._check_threshold(
            sample_1=self.sample_increase_1,
            sample_2=self.sample_increase_2,
            alpha=alpha,
        )

    def check_changes(self) -> Tuple[bool, bool]:
        """Check drift and warning cases.

        :return: drift and warning flags
        :rtype: Tuple[bool, bool]
        """
        drift = self._check_mean_increase(alpha=self.alpha_d)
        warning = False if drift else self._check_mean_increase(alpha=self.alpha_w)
        return drift, warning

    @staticmethod
    def _mcdiarmid_error_bound(
        independent_bound_condition: float, alpha: float
    ) -> float:
        return np.sqrt(independent_bound_condition * np.log(1 / alpha) / 2)

    def reset(self) -> None:
        """Reset sample info variables and increase cut point."""
        self.total = SampleInfo(lambda_=self.lambda_)
        self.sample_increase_1 = SampleInfo(lambda_=self.lambda_)
        self.sample_increase_2 = SampleInfo(lambda_=self.lambda_)
        self.increase_cut_point = float("inf")

    def update_stats(self, value: float, alpha: float) -> None:
        """Update statistics.

        :param value: value to use in update
        :type value: float
        :param alpha: significance value
        :type alpha: float
        """
        self.total.update(value=value)
        epsilon = self._mcdiarmid_error_bound(
            independent_bound_condition=self.total.independent_bound_condition,
            alpha=alpha,
        )
        ewma_plus_epsilon = self.total.ewma.mean + epsilon
        if ewma_plus_epsilon < self.increase_cut_point:
            self.increase_cut_point = ewma_plus_epsilon
            self.sample_increase_1 = copy.deepcopy(self.total)
            self.sample_increase_2 = SampleInfo(lambda_=self.lambda_)
        else:
            self.sample_increase_2.update(value=value)


class McDiarmidTwoSidedTest(McDiarmidOneSidedTest):
    """McDiarmid two-sided test class."""

    def __init__(self, alpha_d: float, alpha_w: float, lambda_: float) -> None:
        """Init method.

        :param alpha_d: significance value for drift
        :type alpha_d: float
        :param alpha_w: significance value for warning
        :type alpha_w: float
        :param lambda_: weight given to recent data compared to older data
        :type lambda_: float
        """
        super().__init__(alpha_d=alpha_d, alpha_w=alpha_w, lambda_=lambda_)
        self.sample_decrease_1 = SampleInfo(lambda_=lambda_)
        self.sample_decrease_2 = SampleInfo(lambda_=lambda_)
        self.decrease_cut_point = float("-inf")

    def _check_mean_decrease(self, alpha: float) -> bool:
        return self._check_threshold(
            sample_1=self.sample_increase_2,
            sample_2=self.sample_increase_1,
            alpha=alpha,
        )

    def check_changes(self) -> Tuple[bool, bool]:
        """Check drift and warning cases.

        :return: drift and warning flags
        :rtype: Tuple[bool, bool]
        """
        drift_increase, warning_increase = super().check_changes()
        drift_decrease = (
            False if drift_increase else self._check_mean_decrease(alpha=self.alpha_d),
        )
        warning_decrease = (
            False
            if warning_increase or drift_decrease
            else self._check_mean_decrease(alpha=self.alpha_w)
        )
        return any([drift_increase, drift_decrease]), any(
            [warning_increase, warning_decrease]
        )

    def reset(self) -> None:
        """Reset sample info variables and cut points."""
        super().reset()
        self.sample_decrease_1 = SampleInfo(lambda_=self.lambda_)
        self.sample_decrease_2 = SampleInfo(lambda_=self.lambda_)
        self.decrease_cut_point = float("-inf")

    def update_stats(self, value: float, alpha: float) -> None:
        """Update statistics.

        :param value: value to use in update
        :type value: float
        :param alpha: significance value
        :type alpha: float
        """
        super().update_stats(value=value, alpha=alpha)
        epsilon = self._mcdiarmid_error_bound(
            independent_bound_condition=self.total.independent_bound_condition,
            alpha=alpha,
        )
        ewma_minus_epsilon = self.total.ewma.mean - epsilon
        if ewma_minus_epsilon > self.decrease_cut_point:
            self.decrease_cut_point = ewma_minus_epsilon
            self.sample_decrease_1 = copy.deepcopy(self.total)
            self.sample_decrease_2 = SampleInfo(lambda_=self.lambda_)
        else:
            self.sample_decrease_2.update(value=value)


class HDDMW(BaseSPC):
    """HDDM-W (Hoeffding's drift detection method with W-Test) [frias2014online]_ detector.

    :References:

    .. [frias2014online] Frias-Blanco, Isvani, et al.
        "Online and non-parametric drift detection methods based on Hoeffding’s bounds."
        IEEE Transactions on Knowledge and Data Engineering 27.3 (2014):
        810-823.
    """

    config_type = HDDMWConfig  # type: ignore

    def __init__(
        self,
        config: Optional[HDDMWConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[HDDMWConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        """
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "test_type": (
                McDiarmidTwoSidedTest(
                    alpha_d=self.config.alpha_d,  # type: ignore
                    alpha_w=self.config.alpha_w,  # type: ignore
                    lambda_=self.config.lambda_,  # type: ignore
                )
                if self.config.two_sided_test  # type: ignore
                else McDiarmidOneSidedTest(
                    alpha_d=self.config.alpha_d,  # type: ignore
                    alpha_w=self.config.alpha_w,  # type: ignore
                    lambda_=self.config.lambda_,  # type: ignore
                )
            ),
            **self.additional_vars,  # type: ignore
        }
        self._set_additional_vars_callback()

    @property
    def test_type(self) -> McDiarmidOneSidedTest:
        """Test type property.

        :return: test type value
        :rtype: McDiarmidOneSidedTest
        """
        return self._additional_vars["test_type"]

    def _update(self, value: Union[int, float], **kwargs) -> None:
        self.num_instances += 1

        self.test_type.update_stats(
            value=value, alpha=self.config.lambda_  # type: ignore
        )

        if self.num_instances >= self.config.min_num_instances:
            drift_flag, warning_flag = self.test_type.check_changes()
            if drift_flag:
                # Out-of-Control
                self.drift = True
                self.warning = False
                self.test_type.reset()
            else:
                if warning_flag:
                    # Warning
                    self.warning = True
                else:
                    # In-Control
                    self.warning = False
                self.drift = False
        else:
            self.drift, self.warning = False, False
