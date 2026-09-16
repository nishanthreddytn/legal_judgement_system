from app.services.categorizer import categorize_case

def test_theft():
    assert categorize_case("The accused was charged with theft of gold")["case_type"] == "Criminal Law"
