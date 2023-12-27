import gettext
import locale
import os

from appdirs import user_config_dir

from graxpert.preferences import load_preferences
from graxpert.resource_utils import resource_path

prefs_file = os.path.join(user_config_dir(appname="GraXpert"), "preferences.json")
prefs = load_preferences(prefs_file)

lang = None
if prefs.lang is None:
    lang, enc = locale.getdefaultlocale()
    if lang is None:
        lang = "en_EN"
    elif lang.startswith("de") or lang.startswith("gsw"):
        lang = "de_DE"
    else:
        lang = "en_EN"

else:
    lang = prefs.lang
    if lang == "Deutsch":
        lang = "de_DE"
    else:
        lang = "en_EN"


lang_gettext = gettext.translation("base", localedir=resource_path("locales"), languages=[lang], fallback=True)
lang_gettext.install()


def _(text):
    return lang_gettext.gettext(text)
