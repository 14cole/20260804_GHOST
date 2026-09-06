"""Finite inverse-design choices without materializing their Cartesian product."""
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
import math


@dataclass(frozen=True)
class NumericChoices:
    start: Decimal
    step: Decimal
    count: int

    @classmethod
    def build(cls, low, high, step, label):
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in (low, high)):
            raise ValueError(f'{label}: limits must be finite and positive.')
        low, high = Decimal(str(low)), Decimal(str(high))
        if high < low:
            raise ValueError(f'{label}: maximum must be at least minimum.')
        if step is not None and (not math.isfinite(float(step)) or float(step) <= 0):
            raise ValueError(f'{label}: step must be finite and positive.')
        if low == high:
            return cls(low, Decimal(0), 1)
        if step is None:
            raise ValueError(f'{label}: set a step to define every allowed value, or select Fixed.')
        step = Decimal(str(step))
        count = int(((high - low) / step).to_integral_value(rounding=ROUND_FLOOR)) + 1
        # Reject a resolution that the floating-point solver cannot distinguish.
        if count > 1 and float(step) < math.ulp(float(high)):
            raise ValueError(f'{label}: step is too small to distinguish adjacent values.')
        return cls(low, step, count)

    def value(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        return float(self.start + index * self.step)


class DesignGrid:
    def __init__(self, layers):
        if not layers:
            raise ValueError('Add at least one layer.')
        self.axes = []
        self.kinds = []
        for index, layer in enumerate(layers, 1):
            low, high, step = ((layer.inv_rs_min, layer.inv_rs_max, layer.inv_rs_accuracy)
                               if layer.is_sheet else
                               (layer.inv_t_min_in, layer.inv_t_max_in, layer.inv_t_accuracy_in))
            if (low is None) != (high is None):
                raise ValueError(f'Layer {index}: set both limits, or select Fixed.')
            if low is None:
                low = high = layer.sheet_resistance if layer.is_sheet else layer.thickness_in
            self.axes.append(NumericChoices.build(low, high, step, f'Layer {index}'))
            self.kinds.append('rs' if layer.is_sheet else 't')
        self.total = math.prod(axis.count for axis in self.axes)

    def design(self, index):
        """Stable mixed-radix ordering; last layer varies fastest. Resume by index."""
        if not 0 <= index < self.total:
            raise IndexError(index)
        thickness = [0.] * len(self.axes)
        resistance = [0.] * len(self.axes)
        for layer in range(len(self.axes) - 1, -1, -1):
            index, choice = divmod(index, self.axes[layer].count)
            target = resistance if self.kinds[layer] == 'rs' else thickness
            target[layer] = self.axes[layer].value(choice)
        return thickness, resistance

    def description(self):
        return '; '.join(
            f'Layer {i}: {axis.count:,} value(s), {axis.value(0):g}–{axis.value(axis.count - 1):g} '
            + ('Ω/sq' if kind == 'rs' else 'in')
            for i, (axis, kind) in enumerate(zip(self.axes, self.kinds), 1))
