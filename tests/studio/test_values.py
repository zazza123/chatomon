from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from chatomon.errors import ChatomonConfigError
from chatomon.studio import _values as values
from chatomon.studio._values import OPAQUE, InvalidSettingError


@dataclass(frozen=True, kw_only=True)
class Address:
    url: str
    token_env: str | None = None
    retries: int = 3
    mode: Literal["fast", "safe"] = "safe"
    tags: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.url.startswith("http"):
            raise ChatomonConfigError("url must start with http")


class Rule(BaseModel):
    title: str
    text: str


class Nothing(BaseModel):
    pass


class Sheet(BaseModel):
    role: str
    notes: str | list[str] | None = None
    rules: str | list[str | Rule] | None = None
    extra: dict[str, int] | None = None


class Odd(BaseModel):
    title: str
    materials: list[Nothing] | None


class Locked(BaseModel):
    title: str
    thing: dict[str, int]


class Client:
    def __init__(self, url: str) -> None:
        self.url = url


class Longer(Address):
    pass


@dataclass(frozen=True, kw_only=True)
class Listed:
    title: str
    tags: list[str] = field(default_factory=list)


class ListedModel(BaseModel):
    title: str
    tags: list[str] = Field(default_factory=list)


@dataclass
class Ping:
    title: str
    pongs: Any = None


@dataclass
class Pong:
    title: str
    pings: list[Ping] | None = None


# Two types that hold each other: the first says so once the second exists.
Ping.__init__.__annotations__["pongs"] = list[Pong] | None


class Chapter(BaseModel):
    title: str
    rules: list[Rule] | None = None


class Book(BaseModel):
    title: str
    chapters: list[Chapter] | None = None


RULE = {"title": "T", "text": "X"}


def block(annotation: Any, *, required: bool = True) -> values.Setting:
    found = values.block("value", annotation, required)
    assert found is not None
    return found


def shapes(cls: type) -> dict[str, tuple[str, bool, bool]]:
    found = values.value_type(cls)
    assert found is not None
    return {f.key: (f.type, f.optional, f.required) for f in found.fields}


def test_a_dataclass_is_read_one_field_at_a_time():
    assert shapes(Address) == {
        "url": ("text", False, True),
        "token_env": ("env", True, False),
        "retries": ("int", False, False),
        "mode": ("choice", False, False),
        "tags": ("lines", True, False),
    }
    found = values.value_type(Address)
    assert found is not None and found.unread == ()
    mode = found.field("mode")
    assert mode is not None and (mode.options, mode.default) == (("fast", "safe"), "safe")


def test_a_pydantic_model_is_read_the_same_way():
    assert shapes(Sheet) == {
        "role": ("text", False, True),
        "notes": ("lines", True, False),
        "rules": ("items", True, False),
    }
    found = values.value_type(Sheet)
    assert found is not None
    rules = found.field("rules")
    assert rules is not None and rules.text is True
    assert [known.cls for known in rules.types] == [Rule]
    # A field the studio does not read is left at its default.
    assert found.unread == (("extra", None, False),)


def test_a_field_without_a_default_that_accepts_none_is_given_none():
    found = values.value_type(Odd)
    assert found is not None
    assert [field.key for field in found.fields] == ["title"]
    assert found.unread == (("materials", None, True),)
    made = values.built(block(Odd), {"type": "Odd", "fields": {"title": "T"}})
    assert made == Odd(title="T", materials=None)


@pytest.mark.parametrize("cls", [Client, Locked, Nothing, str, int, list[str], None])
def test_what_is_not_a_value_type(cls):
    assert values.value_type(cls) is None


def test_a_parameter_that_accepts_a_value_type_is_a_block():
    one = block(Client | Address)
    assert (one.type, one.text, one.optional, one.required) == ("block", False, False, True)
    assert [known.cls for known in one.types] == [Address]
    several = block(str | Sheet | Address | None, required=False)
    assert (several.text, several.optional, several.required) == (True, True, False)
    assert [known.cls for known in several.types] == [Sheet, Address]
    assert values.block("value", str | Client | None, False) is None
    assert values.block("value", int, True) is None


def test_the_schema_of_a_block_carries_its_types():
    schema = block(str | Sheet).schema()
    assert (schema["type"], schema["text"], schema["default"]) == ("block", True, None)
    sheet = schema["types"][0]
    assert sheet["type"] == "Sheet"
    rules = next(field for field in sheet["fields"] if field["key"] == "rules")
    assert (rules["type"], rules["text"]) == ("items", True)
    assert rules["types"][0] == {
        "type": "Rule",
        "fields": [
            {"key": "title", "type": "text", "optional": False, "required": True, "default": None},
            {"key": "text", "type": "text", "optional": False, "required": True, "default": None},
        ],
    }
    assert "types" not in values.Setting("n", "int", False, False, 1).schema()


def test_a_value_is_read_back_with_every_field():
    sheet = Sheet(role="Courier", notes=["a", "b"], rules=[Rule(title="T", text="X"), "plain"])
    assert values.represented(block(str | Sheet), sheet) == {
        "type": "Sheet",
        "fields": {
            "role": "Courier",
            "notes": ["a", "b"],
            "rules": [{"type": "Rule", "fields": {"title": "T", "text": "X"}}, "plain"],
        },
    }
    assert values.represented(block(str | Sheet), "One text.") == "One text."
    assert values.represented(block(str | Sheet | None, required=False), None) is None
    address = Address(url="http://a", token_env="TOKEN")
    assert values.represented(block(Client | Address), address)["fields"] == {
        "url": "http://a",
        "token_env": "TOKEN",
        "retries": 3,
        "mode": "safe",
        "tags": [],
    }


def test_a_text_is_shown_as_its_lines_and_a_sentence_as_one_entry():
    sheet = Sheet(role="r", notes="first\n\n  second  ", rules="Only one.")
    fields = values.represented(block(Sheet), sheet)["fields"]
    assert (fields["notes"], fields["rules"]) == (["first", "second"], ["Only one."])


@pytest.mark.parametrize(
    "value",
    [
        Client("http://a"),
        Longer(url="http://a"),
        Sheet(role="r", extra={"a": 1}),
        Sheet(role="r", notes=["two\nlines"]),
        Sheet(role="r", notes=[" padded "]),
        Sheet(role="r", rules=[Rule(title="T", text="X")]).model_copy(update={"rules": [3]}),
    ],
    ids=["client", "subclass", "unread-field", "line-break", "padding", "odd-entry"],
)
def test_what_the_studio_could_not_write_back_is_opaque(value):
    assert values.represented(block(str | Client | Address | Sheet), value) is OPAQUE


def test_a_text_is_opaque_where_no_text_is_accepted():
    assert values.represented(block(Client | Address), "http://a") is OPAQUE


def test_a_block_is_built_from_its_json():
    made = values.built(
        block(str | Sheet),
        {
            "type": "Sheet",
            "fields": {
                "role": "Courier",
                "notes": ["a", "b"],
                "rules": [{"type": "Rule", "fields": {"title": "T", "text": "X"}}, "plain"],
            },
        },
    )
    assert made == Sheet(
        role="Courier", notes=["a", "b"], rules=[Rule(title="T", text="X"), "plain"]
    )
    assert values.built(block(str | Sheet), "One text.") == "One text."
    assert values.built(block(str | Sheet | None, required=False), None) is None


def test_fields_left_out_take_the_defaults_of_the_type():
    made = values.built(block(Address), {"type": "Address", "fields": {"url": "http://a"}})
    assert made == Address(url="http://a")


def test_an_empty_list_of_an_optional_field_is_none():
    made = values.built(
        block(Sheet), {"type": "Sheet", "fields": {"role": "r", "notes": [], "rules": []}}
    )
    assert (made.notes, made.rules) == (None, None)


def test_an_untouched_field_keeps_the_value_it_has():
    held = Sheet(role="r", notes="first\nsecond", rules="Only one.")
    setting = block(Sheet)
    shown = values.represented(setting, held)
    shown["fields"]["role"] = "Courier"
    made = values.built(setting, shown, held)
    # Shown as lines and as one entry, both are still the texts they were.
    assert (made.role, made.notes, made.rules) == ("Courier", "first\nsecond", "Only one.")
    shown["fields"]["notes"] = ["first", "second", "third"]
    assert values.built(setting, shown, held).notes == ["first", "second", "third"]


def test_the_constructor_of_a_type_refuses_with_its_own_message():
    with pytest.raises(InvalidSettingError) as own:
        values.built(block(Address), {"type": "Address", "fields": {"url": "ftp://a"}})
    assert str(own.value) == "url must start with http"
    with pytest.raises(InvalidSettingError) as pydantic:
        values.built(block(Sheet), {"type": "Sheet", "fields": {"role": "r", "notes": [3]}})
    assert str(pydantic.value).startswith("notes")
    assert "\n" not in str(pydantic.value)


@pytest.mark.parametrize(
    ("annotation", "value", "message"),
    [
        (Client | Address, None, "`value` needs a value"),
        (Client | Address, "http://a", "`value` must be an Address"),
        (Client | Address, 3, "`value` must be an Address"),
        (Client | Address, {"type": "Sheet", "fields": {}}, "`value` must be an Address"),
        (Client | Address, {"type": "Address", "fields": [], "x": 1}, "`value` must be an Address"),
        (str | Sheet | Address, 3, "`value` must be text, a Sheet or an Address"),
        (str | Sheet, "   ", "`value` needs a value"),
        (Address, {"type": "Address", "fields": {}}, "`url` needs a value"),
        (Address, {"type": "Address", "fields": {"url": " "}}, "`url` needs a value"),
        (Address, {"type": "Address", "fields": {"url": 3}}, "`url` must be text"),
        (
            Address,
            {"type": "Address", "fields": {"url": "http://a", "port": 1}},
            "`Address` has no field `port`",
        ),
        (
            Address,
            {"type": "Address", "fields": {"url": "http://a", "retries": "3"}},
            "`retries` must be a whole number",
        ),
        (
            Address,
            {"type": "Address", "fields": {"url": "http://a", "retries": None}},
            "`retries` needs a value",
        ),
        (
            Address,
            {"type": "Address", "fields": {"url": "http://a", "mode": "slow"}},
            "`mode` must be one of: fast, safe",
        ),
        (
            Address,
            {"type": "Address", "fields": {"url": "http://a", "tags": "a"}},
            "`tags` must be a list of texts",
        ),
        (
            Sheet,
            {"type": "Sheet", "fields": {"role": "r", "rules": "one"}},
            "`rules` must be a list of texts and Rule entries",
        ),
        (
            Sheet,
            {"type": "Sheet", "fields": {"role": "r", "rules": [{"type": "Sheet"}]}},
            "`rules` must be a list of texts and Rule entries",
        ),
        (
            Sheet,
            {
                "type": "Sheet",
                "fields": {"role": "r", "rules": [{"type": "Rule", "fields": {"title": "T"}}]},
            },
            "every Rule needs `text`",
        ),
        (
            Sheet,
            {
                "type": "Sheet",
                "fields": {
                    "role": "r",
                    "rules": [{"type": "Rule", "fields": {"title": "T", "text": "X", "n": 1}}],
                },
            },
            "`Rule` has no field `n`",
        ),
        *[
            (
                Address,
                {"type": "Address", "fields": {"url": "http://a", "tags": tags}},
                "every entry of `tags` must be one line of text, with no space around it",
            )
            for tags in (["a "], [" a"], ["a", ""], ["a\nb"])
        ],
    ],
)
def test_a_block_of_the_wrong_shape_is_refused_by_the_studio(annotation, value, message):
    setting = block(annotation)
    with pytest.raises(InvalidSettingError) as caught:
        values.check_shape(setting, value)
    assert str(caught.value) == message
    assert caught.value.setting == "value"


def test_a_block_of_the_right_shape_passes():
    setting = block(str | Sheet | None, required=False)
    for value in (
        None,
        "One text.",
        {"type": "Sheet", "fields": {"role": "r"}},
        {"type": "Sheet", "fields": {"role": "r", "notes": None, "rules": None}},
        {"type": "Sheet", "fields": {"role": "r", "notes": [], "rules": ["plain"]}},
    ):
        values.check_shape(setting, value)


@pytest.mark.parametrize(
    "value",
    [
        "One text.",
        "Two\nlines, and a space after them. ",
        {"type": "Sheet", "fields": {"role": "r"}},
        {"type": "Sheet", "fields": {"role": " r ", "notes": ["first", "second one"]}},
        {
            "type": "Sheet",
            "fields": {
                "role": "r",
                "rules": [" a sentence\nof two lines ", {"type": "Rule", "fields": RULE}],
            },
        },
        {"type": "Address", "fields": {"url": "http://a", "tags": ["a", "b c"], "retries": 0}},
        {"type": "Address", "fields": {"url": "http://a", "token_env": "TOKEN", "mode": "fast"}},
    ],
)
def test_what_the_studio_accepts_it_can_show_and_build_again(value):
    # A value the studio built and cannot show would become an object "from code".
    setting = block(str | Sheet | Address | None, required=False)
    values.check_shape(setting, value)
    made = values.built(setting, value)
    shown = values.represented(setting, made)
    assert shown is not OPAQUE
    assert values.built(setting, shown) == made


@pytest.mark.parametrize("cls", [Listed, ListedModel])
def test_a_default_made_by_a_factory_is_read(cls):
    found = values.value_type(cls)
    assert found is not None
    assert ([f.default for f in found.fields], found.unread) == ([None, []], ())
    setting = block(cls)
    made = values.built(setting, {"type": cls.__name__, "fields": {"title": "t"}})
    shown = {"type": cls.__name__, "fields": {"title": "t", "tags": []}}
    assert values.represented(setting, made) == shown
    assert values.compact(setting, shown) == {"type": cls.__name__, "fields": {"title": "t"}}


def test_a_block_with_a_default_shows_it():
    setting = values.block("value", Address | Client, False, Address(url="http://a"))
    assert setting is not None
    assert setting.schema()["default"] == {
        "type": "Address",
        "fields": {"url": "http://a", "token_env": None, "retries": 3, "mode": "safe", "tags": []},
    }
    # A default the studio cannot write, such as a client, is not shown.
    unwritable = values.block("value", Address | Client, False, Client("http://a"))
    assert unwritable is not None and unwritable.default is None


def test_types_that_hold_each_other_are_read_the_same_whatever_is_read_first():
    def read(cls: type) -> list[tuple[str, str, tuple[tuple[str, ...], ...]]]:
        found = values.value_type(cls)
        assert found is not None
        return [
            (f.key, f.type, tuple(tuple(inner.key for inner in t.fields) for t in f.types))
            for f in found.fields
        ]

    values._outermost.cache_clear()
    ping_first = {cls: read(cls) for cls in (Ping, Pong)}
    values._outermost.cache_clear()
    pong_first = {cls: read(cls) for cls in (Pong, Ping)}
    assert ping_first == pong_first
    # The type that would close the circle is read without the field that closes it.
    assert ping_first[Ping] == [("title", "text", ()), ("pongs", "items", (("title",),))]
    assert ping_first[Pong] == [("title", "text", ()), ("pings", "items", (("title",),))]


def test_an_entry_with_entries_of_its_own_is_left_to_the_code():
    # The page draws the entries of a list, not the lists inside an entry.
    found = values.value_type(Book)
    assert found is not None
    assert [f.key for f in found.fields] == ["title"]
    assert [name for name, _default, _given in found.unread] == ["chapters"]
    setting = block(Book)
    plain = {"type": "Book", "fields": {"title": "t"}}
    assert values.represented(setting, Book(title="t")) == plain
    assert values.represented(setting, Book(title="t", chapters=[Chapter(title="c")])) is OPAQUE


def test_a_field_left_out_keeps_a_held_value_that_reads_as_the_default():
    # In code an empty list where the default is none: a preset leaves the field out,
    # and what is built from it must be the object the code gives.
    setting = block(Sheet)
    held = Sheet(role="r", notes=[])
    edited = {"type": "Sheet", "fields": {"role": "other", "notes": []}}
    changed = values.built(setting, edited, held)
    preset = values.compact(setting, values.represented(setting, changed))
    assert preset == {"type": "Sheet", "fields": {"role": "other"}}
    again = values.built(setting, preset, held)
    assert again == changed and again.notes == []


def test_what_says_where_an_orbital_connects():
    # A setting that names a variable, and a block with a field that does, at any depth.
    assert values.connects(block(Address)) is True
    assert values.connects(block(str | Address | None, required=False)) is True
    assert values.connects(block(Sheet)) is False
    assert values.connects(values.Setting("token_env", "env", True, False, None)) is True
    assert values.connects(values.Setting("reply", "text", True, False, None)) is False


def test_a_preset_leaves_out_the_fields_that_hold_their_default():
    setting = block(Client | Address)
    full = values.represented(setting, Address(url="http://a", token_env="TOKEN"))
    assert values.compact(setting, full) == {
        "type": "Address",
        "fields": {"url": "http://a", "token_env": "TOKEN"},
    }
    sheet = block(str | Sheet)
    held = Sheet(role="r", rules=[Rule(title="T", text="X")])
    assert values.compact(sheet, values.represented(sheet, held)) == {
        "type": "Sheet",
        "fields": {"role": "r", "rules": [{"type": "Rule", "fields": {"title": "T", "text": "X"}}]},
    }
    assert values.compact(sheet, "One text.") == "One text."
    assert values.built(setting, values.compact(setting, full)) == Address(
        url="http://a", token_env="TOKEN"
    )


def test_the_variables_a_block_names_are_found():
    setting = block(Client | Address)
    named = {"type": "Address", "fields": {"url": "http://a", "token_env": "TOKEN"}}
    assert values.variables(setting, named) == ["TOKEN"]
    assert values.variables(setting, {"type": "Address", "fields": {"url": "http://a"}}) == []
    assert values.variables(block(str | Sheet), "One text.") == []
