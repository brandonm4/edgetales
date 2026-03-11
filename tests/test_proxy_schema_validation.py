import unittest

from proxy_server import _validate_json_schema


class ProxySchemaValidationTests(unittest.TestCase):
    def test_nullable_string_accepts_null(self):
        schema = {"type": ["string", "null"]}
        _validate_json_schema(None, schema)

    def test_nullable_object_accepts_null(self):
        schema = {
            "type": ["object", "null"],
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        _validate_json_schema(None, schema)

    def test_nullable_string_rejects_wrong_type(self):
        schema = {"type": ["string", "null"]}
        with self.assertRaisesRegex(ValueError, r"expected string or null"):
            _validate_json_schema(123, schema)


if __name__ == "__main__":
    unittest.main()
