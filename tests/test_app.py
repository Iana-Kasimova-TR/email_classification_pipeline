from app_classification import form_response


class NotEmpty(Exception):
    def __init__(self, message="Text shouldn't be empty"):
        self.message = message
        super().__init__(self.message)


input_data = {
    "incorrect_values": {
        "email_text": "",
    },
    "correct_values": {"email_text": "test email"},
}


def test_form_response_incorrect_values(data=input_data["incorrect_values"]):
    res = form_response(data)
    assert res == NotEmpty().message
