"""NUPS option container for the HOPS Application 1 (Data & Target) settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Mapping, Optional, Union


@dataclass
class NupsOption:
    """In-code representation of the Data & Target configuration values.

    Each attribute mirrors an option that the original HOPS UI exposed. The
    defaults match the values that appear in the interface when the window is
    first shown.
    """

    # Directory selection
    directory: str = "Choose Directory"
    directory_short: str = "Choose Directory"

    # File name patterns
    observation_files: str = "object"
    bias_files: str = "bias"
    dark_files: str = "dark"
    darkf_files: str = "---"
    flat_files: str = "flat"

    # FITS header keywords and timestamp handling
    exposure_time_key: str = "EXPTIME"
    observation_date_key: str = "DATE-OBS"
    observation_time_key: str = "TIME-OBS"
    time_stamp: str = "exposure start"
    master_bias_method: str = "median"
    master_dark_method: str = "median"
    master_darkf_method: str = "median"
    master_flat_method: str = "median"

    # Target selection
    target_ra_dec_choice: int = 0
    auto_target_ra_dec: str = "hh:mm:ss +dd:mm:ss"
    manual_target_ra_dec: str = "hh:mm:ss +dd:mm:ss"
    simbad_target_name: str = "Target name"
    target_ra_dec: str = "hh:mm:ss +dd:mm:ss"
    target_name: str = "Your target"

    # Location selection
    location_choice: int = 0
    auto_location: str = "+dd:mm:ss +dd:mm:ss"
    profile_location: str = "+dd:mm:ss +dd:mm:ss"
    manual_location: str = "+dd:mm:ss +dd:mm:ss"
    location: str = "+dd:mm:ss +dd:mm:ss"
    location_inited: bool = False

    # Instrument filter setting
    filter: str = "No filter chosen"
    filter_key: str = "FILTER"

    # Advanced settings
    bin_fits: int = 1
    crop_edge_pixels: int = 0
    crop_x1: int = 0
    crop_x2: int = 0
    crop_y1: int = 0
    crop_y2: int = 0
    faint_target_mode: bool = False
    moving_target_mode: bool = False
    rotating_field_mode: bool = False
    colour_camera_mode: bool = False
    centroids_snr: int = 3
    psf_guess: int = 2
    stars_snr: int = 4
    reduction_prefix: str = "out_"
    reduction_directory: str = "REDUCED_DATA"

    # Observer information
    observer: str = ""
    observatory: str = "My Observatory"
    telescope: str = ""
    camera: str = ""
    observer_key: str = "OBSERVER"
    observatory_key: str = "OBSERVAT"
    telescope_key: str = "TELESCOP"
    camera_key: str = "INSTRUME"

    # Observatory profile data
    observatory_latitude_key: str = "SITELAT,LAT-OBS,LATITUDE"
    observatory_longitude_key: str = "SITELONG,LONG-OBS,LONGITUD"
    observatory_lat: str = ""
    observatory_long: str = ""
    target_ra_key: str = "OBJCTRA,RA"
    target_dec_key: str = "OBJCTDEC,DEC"

    # Workflow state flags
    data_target_complete: bool = False
    data_target_version: Union[bool, str] = False
    proceed: bool = False

    def to_dict(self) -> Dict[str, Union[str, int, bool]]:
        """Return the options as a plain dictionary."""

        return asdict(self)

    def update_location(self, fits_obj: Any) -> "NupsOption":
        """Return a copy of this option set updated from a FITS header.

        Parameters
        ----------
        fits_obj
            Either a :class:`dict` representing a FITS header or any object with
            a ``header`` attribute (for example :class:`~nups_core.util.FitsFrame`).

        Returns
        -------
        NupsOption
            A new instance with location and target information populated from
            the header. If the required keys are missing, the original instance
            is returned unchanged.
        """

        header: Optional[Mapping[str, Any]]
        if isinstance(fits_obj, Mapping):
            header = fits_obj
        else:
            header = getattr(fits_obj, "header", None)
            if header is not None and not isinstance(header, Mapping):
                raise TypeError("fits_obj.header must be a mapping")

        if header is None:
            raise TypeError("fits_obj must be a mapping or expose a 'header' mapping")

        latitude = _first_header_value(header, self.observatory_latitude_key)
        longitude = _first_header_value(header, self.observatory_longitude_key)
        ra_value = _first_header_value(header, self.target_ra_key)
        dec_value = _first_header_value(header, self.target_dec_key)
        object_name = header.get("OBJECT") or header.get("TARGET")
        observation_date_value = header.get(self.observation_date_key)
        observation_time_value = header.get(self.observation_time_key)

        updates: Dict[str, Any] = {}
        location_updated = False

        if latitude is not None and longitude is not None:
            location_str = f"{_stringify_header_value(latitude)} {_stringify_header_value(longitude)}"
            updates.update(
                {
                    "location": location_str,
                    "auto_location": location_str,
                }
            )
            location_updated = True

        if ra_value is not None and dec_value is not None:
            target_coord = f"{_stringify_header_value(ra_value)} {_stringify_header_value(dec_value)}"
            updates.update(
                {
                    "target_ra_dec": target_coord,
                    "auto_target_ra_dec": target_coord,
                }
            )

        if object_name:
            updates["target_name"] = _stringify_header_value(object_name)

        if observation_time_value is None and observation_date_value is not None:
            observation_time_str = _stringify_header_value(observation_date_value)
            if "T" in observation_time_str and self.observation_time_key != self.observation_date_key:
                updates["observation_time_key"] = self.observation_date_key

        if not updates:
            return self

        updates["location_inited"] = location_updated or self.location_inited
        return replace(self, **updates)


def _first_header_value(header: Mapping[str, Any], keys: str) -> Optional[Any]:
    for key in keys.split(','):
        candidate = key.strip()
        if not candidate:
            continue
        if candidate in header:
            value = header[candidate]
            if value not in (None, ""):
                return value
    return None


def _stringify_header_value(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8').strip()
        except UnicodeDecodeError:
            return value.decode('latin-1', errors='ignore').strip()
    return str(value).strip()
