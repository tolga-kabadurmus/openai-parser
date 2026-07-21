class GenAINormalizerError(Exception): pass
class ProviderDetectionError(GenAINormalizerError): pass
class ParserNotRegisteredError(GenAINormalizerError): pass
class UnsupportedEndpointError(GenAINormalizerError): pass
class NormalizationError(GenAINormalizerError): pass
