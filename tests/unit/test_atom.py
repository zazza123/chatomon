import pytest

from chatomon import Atom, ChatomonConfigError, Knowledge, TurnReport, Verdict
from chatomon.connectors import Connector
from chatomon.engine import Engine
from chatomon.orbitals.base import Orbital
from chatomon.schemas import OrbitalContext
from tests.fakes import FakeAgent, FakeConnector, InputOnlyOrbital, ScriptedOrbital


def make_atom(**overrides) -> Atom[FakeAgent]:
    settings = {
        "name": "speedypack",
        "instructions": "You are the SpeedyPack assistant.",
        "connector": FakeConnector(),
    }
    settings.update(overrides)
    return Atom(**settings)


class BindRecorder(Orbital):
    calls_model = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bound: list[Connector] = []

    def bind(self, connector: Connector) -> None:
        self.bound.append(connector)

    async def check(self, ctx: OrbitalContext) -> Verdict:
        return Verdict.allow()


def test_build_returns_the_connector_agent_with_a_configured_engine():
    connector = FakeConnector()
    first = ScriptedOrbital(name="first")
    second = ScriptedOrbital(name="second")
    atom = make_atom(
        connector=connector,
        input_orbitals=[first],
        output_orbitals=[second],
        max_retries=1,
        refusal="No.",
    )
    agent = atom.build()
    assert connector.built == [agent]
    assert agent.atom is atom
    assert isinstance(agent.engine, Engine)
    assert agent.engine.input_orbitals == (first,)
    assert agent.engine.output_orbitals == (second,)
    assert agent.engine.max_retries == 1
    assert agent.engine.refusal == "No."
    assert agent.engine.atom_name == "speedypack"


def test_system_prompt_appends_the_knowledge():
    assert make_atom().system_prompt == "You are the SpeedyPack assistant."
    atom = make_atom(knowledge="Open 9-18.")
    assert isinstance(atom.knowledge, Knowledge)
    assert atom.system_prompt == (
        "You are the SpeedyPack assistant.\n\n<knowledge>\nOpen 9-18.\n</knowledge>"
    )


def test_bind_is_called_with_the_atom_connector():
    connector = FakeConnector()
    orb = BindRecorder(name="rec")
    make_atom(connector=connector, input_orbitals=[orb]).build()
    assert orb.bound == [connector]


def test_same_instance_in_both_chains_is_allowed_and_bound_once():
    orb = BindRecorder(name="both")
    agent = make_atom(input_orbitals=[orb], output_orbitals=[orb]).build()
    assert len(orb.bound) == 1
    assert agent.engine.input_orbitals == (orb,)
    assert agent.engine.output_orbitals == (orb,)


@pytest.mark.parametrize(
    ("chains", "match"),
    [
        (
            {
                "input_orbitals": [ScriptedOrbital(name="x")],
                "output_orbitals": [ScriptedOrbital(name="x")],
            },
            "two different orbitals are named 'x'",
        ),
        (
            {"input_orbitals": [ScriptedOrbital(name="x"), ScriptedOrbital(name="y", reads=["z"])]},
            "reads 'z'",
        ),
        (
            {"input_orbitals": [ScriptedOrbital(name="y", reads=["x"]), ScriptedOrbital(name="x")]},
            "reads 'x'",
        ),
        (
            {"input_orbitals": [ScriptedOrbital(name="x", reads=["x"])]},
            "its own annotations",
        ),
        (
            {"output_orbitals": [InputOnlyOrbital(name="scope")]},
            "cannot run on the output side",
        ),
    ],
)
def test_invalid_chains_are_rejected_at_build(chains, match):
    with pytest.raises(ChatomonConfigError, match=match):
        make_atom(**chains).build()


def test_the_same_instance_twice_in_one_chain_is_rejected():
    orb = ScriptedOrbital(name="x")
    with pytest.raises(ChatomonConfigError, match="appears twice in input_orbitals"):
        make_atom(input_orbitals=[orb, orb]).build()


def test_output_orbitals_may_read_input_orbitals():
    atom = make_atom(
        input_orbitals=[ScriptedOrbital(name="intents")],
        output_orbitals=[ScriptedOrbital(name="judge", reads=["intents"])],
    )
    atom.validate()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"name": " "}, "name"),
        ({"instructions": ""}, "instructions"),
        ({"connector": object()}, "Connector"),
        ({"input_orbitals": [object()]}, "Orbital"),
        ({"max_retries": -1}, "max_retries"),
        ({"capture_content": "all"}, "capture_content"),
        ({"refusal": ""}, "refusal"),
    ],
)
def test_invalid_settings_are_rejected_at_construction(overrides, match):
    with pytest.raises(ChatomonConfigError, match=match):
        make_atom(**overrides)


async def test_engine_uses_on_turn_and_capture_content():
    received: list[TurnReport] = []
    agent = make_atom(on_turn=received.append, capture_content="hash").build()
    engine = agent.engine
    turn = engine.start_turn()
    await engine.run_input(turn, message="hi")
    await engine.run_output(turn, reply="hello")
    await engine.finish_turn(turn)
    assert received[0].user_message is not None
    assert received[0].user_message.startswith("sha256:")
