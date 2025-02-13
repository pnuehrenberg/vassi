from .annotated import AnnotatedMixin
from .sampleable import SampleableMixin


class AnnotatedSampleableMixin(SampleableMixin, AnnotatedMixin):
    ...
    # for inheritance and typing purposes
