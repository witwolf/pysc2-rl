from pysc2.lib import units
import enum

class UnitSize(enum.IntEnum):
    '''units' radius'''
    Nexus=10
    Pylon=4
    Assimilator=6
    Gateway=7
    CyberneticsCore=7
    PylonPower=23
    Stalker=2

class Unit:
    def __init__(self, id,
                 build_type,
                 unit_type,
                 requirement_types,
                 minerals,
                 gas, time, food):
        self.id = id
        self.build_type = build_type
        self.unit_type = unit_type
        self.requirement_types = requirement_types
        self.minerals = minerals
        self.gas = gas
        self.time = time
        self.food = food


class Building:
    def __init__(self, id,
                 build_type, unit_type, requirement_types,
                 minerals, gas, time, trainable):
        self.id = id
        self.build_type = build_type
        self.unit_type = unit_type
        self.requirement_types = requirement_types
        self.minerals = minerals
        self.gas = gas
        self.time = time
        self.trainable = trainable


# protoss units
Probe = Unit(
    id=0,
    build_type=units.Protoss.Nexus,
    unit_type=units.Protoss.Probe,
    requirement_types=[], minerals=50, gas=0,
    time=68, food=1)

Zealot = Unit(
    id=1,
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.Zealot,
    requirement_types=[],
    minerals=100,
    gas=0, time=152, food=2)

Stalker = Unit(
    id=2,
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.Stalker,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=125, gas=50,
    time=168, food=2)

Sentry = Unit(
    id=3,
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.Sentry,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=50, gas=100,
    time=26, food=2)

Adept = Unit(
    id=4,
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.Adept,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=100, gas=25,
    time=27, food=2)

HighTemplar = Unit(
    id=5,
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.HighTemplar,
    requirement_types=[units.Protoss.TemplarArchive],
    minerals=50, gas=150,
    time=39, food=2)

DarkTemplar = Unit(
    id=6,
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.DarkTemplar,
    requirement_types=[units.Protoss.DarkShrine],
    minerals=125, gas=125,
    time=39, food=2)

Archon = Unit(
    id=7,
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.DarkTemplar,
    requirement_types=[], minerals=0,
    gas=0, time=9, food=4)

Observer = Unit(
    id=8,
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.Observer,
    requirement_types=[],
    minerals=25, gas=75, time=21, food=1)

WarpPrism = Unit(
    id=9,
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.WarpPrism,
    requirement_types=[],
    minerals=200, gas=0, time=36, food=2)

Immortal = Unit(
    id=10,
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.Immortal, requirement_types=[],
    minerals=250, gas=100,
    time=39, food=4)

Colossus = Unit(
    id=11,
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.Colossus,
    requirement_types=[units.Protoss.RoboticsBay],
    minerals=300, gas=200,
    time=54, food=6)

Disruptor = Unit(
    id=12,
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.Disruptor,
    requirement_types=[units.Protoss.RoboticsBay],
    minerals=150, gas=150,
    time=36, food=3)

Phoenix = Unit(
    id=13,
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.Phoenix,
    requirement_types=[], minerals=150,
    gas=100, time=25, food=2)

VoidRay = Unit(
    id=14,
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.VoidRay,
    requirement_types=[], minerals=250,
    gas=150, time=43, food=4)

Oracle = Unit(
    id=15,
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.Oracle,
    requirement_types=[], minerals=150,
    gas=150, time=37, food=3)

Tempest = Unit(
    id=16,
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.Tempest,
    requirement_types=[units.Protoss.FleetBeacon],
    minerals=300, gas=200,
    time=43, food=6)

Carrier = Unit(
    id=17,
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.Carrier,
    requirement_types=[units.Protoss.FleetBeacon],
    minerals=350, gas=250, time=86, food=6)

Interceptor = Unit(
    id=18,
    build_type=units.Protoss.Carrier,
    unit_type=units.Protoss.Interceptor,
    requirement_types=[],
    minerals=15, gas=0, time=6, food=0)

Mothership = Unit(
    id=19,
    build_type=units.Protoss.Nexus,
    unit_type=units.Protoss.Mothership,
    requirement_types=[units.Protoss.FleetBeacon],
    minerals=400, gas=400, time=114, food=8)

# protoss buildings
Pylon = Building(
    id=0,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Pylon,
    requirement_types=[], minerals=100,
    gas=0, time=100, trainable=False)

Gateway = Building(
    id=1,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Gateway,
    requirement_types=[units.Protoss.Nexus, units.Protoss.Pylon],
    minerals=150, gas=0, time=260, trainable=True)

Assimilator = Building(
    id=2,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Assimilator,
    requirement_types=[],
    minerals=75, gas=0, time=120, trainable=False)

Forge = Building(
    id=3,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Forge,
    requirement_types=[units.Protoss.Nexus],
    minerals=150, gas=0, time=32, trainable=False)

CyberneticsCore = Building(
    id=4,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.CyberneticsCore,
    requirement_types=[units.Protoss.Gateway],
    minerals=150, gas=0, time=200, trainable=False)

Nexus = Building(
    id=5,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Nexus,
    requirement_types=[], minerals=400,
    gas=0, time=71, trainable=True)

WarpGate = Building(
    id=6,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.WarpGate,
    requirement_types=[units.Protoss.Gateway],
    minerals=0, gas=0, time=7, trainable=True)

PhotonCannon = Building(
    id=7,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.PhotonCannon,
    requirement_types=[units.Protoss.Forge],
    minerals=150, gas=0, time=29, trainable=False)

ShieldBattery = Building(
    id=8,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.ShieldBattery,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=100, gas=0, time=29, trainable=False)

RoboticsFacility = Building(
    id=9,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.RoboticsFacility,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=200, gas=100, time=46, trainable=True)

Stargate = Building(
    id=10,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Stargate,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=150, gas=100, time=43, trainable=True)

TwilightCouncil = Building(
    id=11,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.TwilightCouncil,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=150, gas=100, time=36, trainable=False)

RoboticsBay = Building(
    id=12,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.RoboticsBay,
    requirement_types=[units.Protoss.RoboticsFacility],
    minerals=200, gas=200, time=46, trainable=False)

FleetBeacon = Building(
    id=13,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.FleetBeacon,
    requirement_types=[units.Protoss.Stargate],
    minerals=300, gas=200, time=43, trainable=False)

TemplarArchive = Building(
    id=14,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.TemplarArchive,
    requirement_types=[units.Protoss.TwilightCouncil],
    minerals=150, gas=200, time=36, trainable=False)

DarkShrine = Building(
    id=15,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.DarkShrine,
    requirement_types=[units.Protoss.TwilightCouncil],
    minerals=150, gas=150, time=71, trainable=False)

StasisTrap = Building(
    id=16,
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.StasisTrap,
    requirement_types=[],
    minerals=0, gas=0, time=4, trainable=False)
