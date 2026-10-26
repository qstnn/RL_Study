import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from research_common import launch
from rounded_cart import replace_wheels


def test_cart_geometry_keeps_true_mass_passive_axles_and_body_clearance(tmp_path):
    original=tmp_path/'original.urdf';variant=tmp_path/'variant.urdf'
    launch.CART.write_cart_urdf(original);before=original.read_bytes();variant.write_bytes(before)
    info=replace_wheels(variant);root=ET.parse(variant).getroot()
    assert original.read_bytes()==before
    assert sum(float(l.find('./inertial/mass').get('value')) for l in root.findall('link'))==40
    for joint in root.findall('joint'):
        assert joint.get('type')=='continuous'
        assert float(joint.find('limit').get('effort'))==0
        x,y,z=map(float,joint.find('origin').get('xyz').split())
        assert abs(y)-.14>.25
    for wheel in root.findall('link')[1:]:
        assert float(wheel.find('./collision/geometry/sphere').get('radius'))==.14
        assert float(wheel.find('./inertial/inertia').get('iyy'))==pytest.approx(.4*.14**2)
    assert info['overall_width_m']==pytest.approx(2*(.4+.14))
