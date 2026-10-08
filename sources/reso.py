"""ResoWebApiSource: STUB. Not implemented on purpose.

A real RESO Web API / IDX feed requires a data license that your broker (and the MLS) must
approve. Until you have written approval and credentials, this class refuses to run rather than
guess endpoints or credentials. See docs/RESO_WEB_API.md for what to ask your broker.
"""

from __future__ import annotations

from collections.abc import Iterator

from sources.base import ListingSource, RawListing

NOT_LICENSED_MESSAGE = (
    "Live MLS feed access is not set up. It needs a data license approved by your broker and "
    "your MLS (for FMLS, via the FMLS Marketplace). Until then, import a CSV export or add "
    "listings by address. See docs/RESO_WEB_API.md for the steps to ask your broker about."
)


class ResoWebApiSource(ListingSource):
    name = "reso"

    def __init__(self, base_url: str | None = None, access_token: str | None = None) -> None:
        # Intentionally no defaults: endpoints and tokens come only from your approved license.
        self.base_url = base_url
        self.access_token = access_token

    def fetch(self) -> Iterator[RawListing]:
        raise NotImplementedError(NOT_LICENSED_MESSAGE)
