IRIS_SCHEMA = {
    "$schema": "http://json-schema.org/draft/2019-09/hyper-schema",
    "title": "Iris Data Schema",
    "description": "Schema of Bezdek Iris data",
    "type": "object",
    "properties": {
        "sepal_length": {
            "type": "number",
            "description": "Sepal length in cm",
            "minimum": 0,
        },
        "sepal_width": {
            "type": "number",
            "description": "Sepal width in cm",
            "minimum": 0,
        },
        "petal_length": {
            "type": "number",
            "description": "Petal length in cm",
            "minimum": 0,
        },
        "petal_width": {
            "type": "number",
            "description": "Petal width in cm",
            "minimum": 0,
        },
        "species": {
            "type": "string",
            "description": "class",
            "enum": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        },
    },
    "required": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ],
    "additionalProperties": False,
}
