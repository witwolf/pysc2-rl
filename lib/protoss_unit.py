from pysc2.lib import units


class Unit:
    def __init__(self, build_type,
                 unit_type,
                 requirement_types,
                 minerals,
                 gas, time, food):
        self.build_type = build_type
        self.unit_type = unit_type
        self.requirement_types = requirement_types
        self.minerals = minerals
        self.gas = gas
        self.time = time
        self.food = food


class Building:
    def __init__(self, build_type,
                 unit_type, requirement_types,
                 minerals, gas, time):
        self.build_type = build_type
        self.unit_type = unit_type
        self.requirement_types = requirement_types
        self.minerals = minerals
        self.gas = gas
        self.time = time


# protoss units
Probe = Unit(
    build_type=units.Protoss.Nexus,
    unit_type=units.Protoss.Probe,
    requirement_types=[], minerals=50, gas=0,
    time=68, food=1)

Zealot = Unit(
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.Zealot,
    requirement_types=[],
    minerals=100,
    gas=0, time=152, food=2)

Stalker = Unit(
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.Stalker,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=125, gas=50,
    time=168, food=2)

Sentry = Unit(
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.Sentry,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=50, gas=100,
    time=26, food=2)

Adept = Unit(
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.Adept,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=100, gas=25,
    time=27, food=2)

HighTemplar = Unit(
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.HighTemplar,
    requirement_types=[units.Protoss.TemplarArchive],
    minerals=50, gas=150,
    time=39, food=2)

DarkTemplar = Unit(
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.DarkTemplar,
    requirement_types=[units.Protoss.DarkShrine],
    minerals=125, gas=125,
    time=39, food=2)

Archon = Unit(
    build_type=units.Protoss.Gateway,
    unit_type=units.Protoss.DarkTemplar,
    requirement_types=[], minerals=0,
    gas=0, time=9, food=4)

Observer = Unit(
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.Observer,
    requirement_types=[],
    minerals=25, gas=75, time=21, food=1)

WarpPrism = Unit(
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.WarpPrism,
    requirement_types=[],
    minerals=200, gas=0, time=36, food=2)

Immortal = Unit(
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.Immortal, requirement_types=[],
    minerals=250, gas=100,
    time=39, food=4)
Colossus = Unit(
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.Colossus,
    requirement_types=[units.Protoss.RoboticsBay],
    minerals=300, gas=200,
    time=54, food=6)

Disruptor = Unit(
    build_type=units.Protoss.RoboticsFacility,
    unit_type=units.Protoss.Disruptor,
    requirement_types=[units.Protoss.RoboticsBay],
    minerals=150, gas=150,
    time=36, food=3)

Phoenix = Unit(
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.Phoenix,
    requirement_types=[], minerals=150,
    gas=100, time=25, food=2)

VoidRay = Unit(
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.VoidRay,
    requirement_types=[], minerals=250,
    gas=150, time=43, food=4)

Oracle = Unit(
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.Oracle,
    requirement_types=[], minerals=150,
    gas=150, time=37, food=3)

Tempest = Unit(
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.Tempest,
    requirement_types=[units.Protoss.FleetBeacon],
    minerals=300, gas=200,
    time=43, food=6)

Carrier = Unit(
    build_type=units.Protoss.Stargate,
    unit_type=units.Protoss.Carrier,
    requirement_types=[units.Protoss.FleetBeacon],
    minerals=350, gas=250, time=86, food=6)

Interceptor = Unit(
    build_type=units.Protoss.Carrier,
    unit_type=units.Protoss.Interceptor,
    requirement_types=[],
    minerals=15, gas=0, time=6, food=0)

Mothership = Unit(
    build_type=units.Protoss.Nexus,
    unit_type=units.Protoss.Mothership,
    requirement_types=[units.Protoss.FleetBeacon],
    minerals=400, gas=400, time=114, food=8)

# protoss buildings
Nexus = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Nexus,
    requirement_types=[], minerals=400,
    gas=0, time=71)

Pylon = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Pylon,
    requirement_types=[], minerals=100,
    gas=0, time=100)

Assimilator = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Assimilator,
    requirement_types=[],
    minerals=75, gas=0, time=120)

Gateway = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Gateway,
    requirement_types=[units.Protoss.Nexus, units.Protoss.Pylon],
    minerals=150, gas=0, time=260)

WarpGate = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.WarpGate,
    requirement_types=[units.Protoss.Gateway],
    minerals=0, gas=0, time=7)

Forge = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Forge,
    requirement_types=[units.Protoss.Nexus],
    minerals=150, gas=0, time=32)

CyberneticsCore = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.CyberneticsCore,
    requirement_types=[units.Protoss.Gateway],
    minerals=150, gas=0, time=200)

PhotonCannon = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.PhotonCannon,
    requirement_types=[units.Protoss.Forge],
    minerals=150, gas=0, time=29)

ShieldBattery = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.ShieldBattery,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=100, gas=0, time=29)

RoboticsFacility = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.RoboticsFacility,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=200, gas=100, time=46)

Stargate = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.Stargate,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=150, gas=100, time=43)

TwilightCouncil = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.TwilightCouncil,
    requirement_types=[units.Protoss.CyberneticsCore],
    minerals=150, gas=100, time=36)

RoboticsBay = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.RoboticsBay,
    requirement_types=[units.Protoss.RoboticsFacility],
    minerals=200, gas=200, time=46)

FleetBeacon = Building(
    build_type=units.Protoss.Probe,

    unit_type=units.Protoss.FleetBeacon,
    requirement_types=[units.Protoss.Stargate],
    minerals=300, gas=200, time=43)

TemplarArchive = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.TemplarArchive,
    requirement_types=[units.Protoss.TwilightCouncil],
    minerals=150, gas=200, time=36)

DarkShrine = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.DarkShrine,
    requirement_types=[units.Protoss.TwilightCouncil],
    minerals=150, gas=150, time=71)

StasisTrap = Building(
    build_type=units.Protoss.Probe,
    unit_type=units.Protoss.StasisTrap,
    requirement_types=[],
    minerals=0, gas=0, time=4)
