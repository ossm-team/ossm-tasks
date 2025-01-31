import xmlschema

def parse_stef_xml(model_stef_xml: str):
    schema = xmlschema.XMLSchema("STEF.xsd")

    try:
        model = schema.to_dict(model_stef_xml)
    except xmlschema.XMLSchemaValidationError as e:
        raise ValueError(f"The model description does not follow the description language: {e.reason}")

    return model