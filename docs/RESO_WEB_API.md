# Live MLS feed (RESO Web API): not implemented, on purpose

`sources/reso.py` contains `ResoWebApiSource`, a stub that raises an explanatory error. A real
feed needs a **data license that your broker and your MLS approve**. Nothing in this project
guesses endpoints, dataset names, or credentials. Until you have written approval, use the CSV
import or add listings by address.

Everything below was checked on 2026-10-08. Details are in [RESEARCH.md](RESEARCH.md) section 1.

## What FMLS publishes (verified)

- FMLS distributes data through **Bridge Interactive**: the native Bridge API, a **RESO Web API**,
  and **vRETS**. [FMLS Marketplace info](https://fmls.com/marketplace-info)
- Third parties are approved through the **FMLS Marketplace**. A vendor creates a *Product*, and
  once approved it shows as "Approved" to FMLS members.
- Fees listed: a $250 application fee (waived for member brokerages) and a monthly fee based on
  the number of subscribed members. The fee schedule on the pricing page is dated 2019, so
  confirm current numbers: [pricing](https://www.fmls.com/marketplace-pricing/)
- Typical review turnaround: 2 business days. Contact: Data@FMLS.com, 678-904-0446.

## Not verified (ask, don't assume)

- Whether an individual agent can register as a vendor, or must go through the brokerage.
- Which data scope covers photos and showing instructions, and whether running ML analysis on
  photos is a permitted use.
- Retention rules (how long you may keep data, what to delete when a listing goes off market).
- The Bridge sign-up flow, auth model, and dataset names: the Bridge documentation page did not
  render when I checked, so read it yourself:
  [Signing up with Bridge API](https://bridgedataoutput.com/docs/platform/Introduction/Signing-up-with-Bridge-API)

## What to ask your broker

1. May I store MLS listing data and photos locally for my own business use? For how long, and
   what must I delete when a listing goes off market?
2. May I run automated analysis (ML scoring) on listing photos?
3. Will the brokerage register a Product on the FMLS Marketplace, or may I register as a vendor?
4. Which data scope applies (IDX-style display, broker back-office, other)? Does it include photo
   URLs and showing instructions?
5. Who signs the data agreement, and who holds the credentials?
6. Can derived scores (my condition scores) be shared with clients, or are they internal only?
7. Does the brokerage have rules about third-party AI tools touching MLS content?

## How it will plug in later

`ResoWebApiSource.fetch()` only has to yield `RawListing` objects (see `sources/base.py`); the
rest of the app (de-duplication, geocoding, photos, scoring, routing) is unchanged. RESO Data
Dictionary field names such as `StreetNumber`, `ListPrice`, `BedroomsTotal` and
`ShowingRequirements` are already recognized by the importer's column detection, so a feed that
is exported or saved as a table works today through the normal file import.
