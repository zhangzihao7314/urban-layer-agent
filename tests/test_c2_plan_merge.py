from src.agent.c2_dialogue import merge_confirmed_percentage_proposal


def _values(**changes):
    names = [
        "TCH", "IMD", "BH", "BSF", "SVF", "F_AC",
        "F_S", "F_M", "F_BS", "F_G", "F_TV", "F_W",
    ]
    return {name: changes.get(name, 0.0) for name in names}


def test_confirming_second_polygon_keeps_first_polygon_plan():
    confirmed, unchanged, predictors = merge_confirmed_percentage_proposal(
        {}, [7, 8, 9], [7, 8, 9],
        {"polygon_id": 8, "percentages": _values(TCH=22, IMD=-11, F_W=33)},
    )
    confirmed, unchanged, predictors = merge_confirmed_percentage_proposal(
        confirmed, unchanged, [7, 8, 9],
        {"polygon_id": 7, "percentages": _values(TCH=15, IMD=-10, F_W=20)},
    )

    assert confirmed[8]["F_W"] == 33
    assert confirmed[7]["F_W"] == 20
    assert unchanged == [9]
    assert predictors == ["TCH", "IMD", "F_W"]


def test_confirming_same_polygon_replaces_only_that_polygon():
    original = {
        7: _values(F_TV=20),
        8: _values(F_W=33),
    }
    confirmed, unchanged, predictors = merge_confirmed_percentage_proposal(
        original, [9], [7, 8, 9],
        {"polygon_id": 7, "percentages": _values(F_G=-10)},
    )

    assert confirmed[7]["F_TV"] == 0
    assert confirmed[7]["F_G"] == -10
    assert confirmed[8]["F_W"] == 33
    assert unchanged == [9]
    assert predictors == ["F_G", "F_W"]
