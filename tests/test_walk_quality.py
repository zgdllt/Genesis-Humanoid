from gs_agent.utils.walk_benchmark import quality_matches


def test_survival_and_tracking_are_required_even_if_return_matches() -> None:
    reference = {
        "return": 100.0,
        "survival": 1.0,
        "velocity_rmse": 0.1,
        "yaw_rmse": 0.1,
        "foot_slip": 0.02,
    }
    assert quality_matches(reference, reference)
    for key, bad in (
        ("return", 90.0),
        ("survival", 0.9),
        ("velocity_rmse", 0.3),
        ("yaw_rmse", 0.4),
        ("foot_slip", 0.2),
    ):
        assert not quality_matches({**reference, key: bad}, reference)


def test_weak_reference_does_not_lower_absolute_quality() -> None:
    weak = {"return": 1.0, "survival": 0.5, "velocity_rmse": 1.0, "yaw_rmse": 1.0, "foot_slip": 0.2}
    assert not quality_matches(weak, weak)
