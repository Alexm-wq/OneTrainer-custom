"""T-LoRA Gradient Only schedule: the min rank over the run, as a percentage of the LoRA rank per epoch.

"0:100, 3:80, 10:20" keeps every rank until epoch 0 (no mask), falls linearly to 80 % of the rank
at epoch 3 and to 20 % at epoch 10, and stays there. Before the first point its value holds, after
the last point the last value. Epochs may be fractional.
"""


def parse_tlora_schedule(text: str) -> list[tuple[float, float]]:
    """(epoch, percent) points, rising in epoch; an empty text is an empty schedule."""
    points = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            epoch, percent = (float(part) for part in item.split(":"))
        except ValueError:
            raise ValueError(f"T-LoRA Schedule: '{item}' is not 'epoch:percent', e.g. '0:100, 3:80, 10:20'.") from None
        if epoch < 0 or not 0 < percent <= 100:
            raise ValueError(f"T-LoRA Schedule: '{item}' needs an epoch of at least 0 and a percentage in (0, 100].")
        if points and epoch <= points[-1][0]:
            raise ValueError("T-LoRA Schedule: the epochs must rise from point to point.")
        points.append((epoch, percent))
    return points


def scheduled_min_rank(points: list[tuple[float, float]], epoch: float, rank: int) -> float:
    """The min rank at a (fractional) epoch, at least 1."""
    if epoch <= points[0][0]:
        percent = points[0][1]
    elif epoch >= points[-1][0]:
        percent = points[-1][1]
    else:
        for (start, low), (end, high) in zip(points, points[1:], strict=False):
            if epoch <= end:
                percent = low + (high - low) * (epoch - start) / (end - start)
                break
    return max(1.0, rank * percent / 100)
