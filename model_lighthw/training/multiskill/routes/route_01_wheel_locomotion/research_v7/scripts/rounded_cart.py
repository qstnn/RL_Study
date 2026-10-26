"""研究用圆弧胎面被动轮；保留40kg总质量、轮半径和完全无驱动。

明确的拖车几何变体：以球面胎面替代原平端圆柱，轮心外移避免轮胎穿箱体。
它不是原圆柱拖车的验收结果；机器人本体、质量和电机完全不改。
"""
from pathlib import Path
import xml.etree.ElementTree as ET


def replace_wheels(path):
    path=Path(path);tree=ET.parse(path);root=tree.getroot()
    if root.get('name')!='r1_passive_cart':raise ValueError('Unexpected cart asset')
    radius=.14;centers=[]
    for index in range(4):
        link=root.find(f"./link[@name='cart_wheel_{index}']");joint=root.find(f"./joint[@name='cart_wheel_{index}_joint']")
        if link is None or joint is None or joint.get('type')!='continuous' or float(joint.find('limit').get('effort'))!=0:raise ValueError('Expected passive original wheel')
        mass=float(link.find('./inertial/mass').get('value'))
        if mass!=1.:raise ValueError('Expected original wheel mass')
        for kind in ('visual','collision'):
            geom=link.find(f'./{kind}/geometry');old=geom.find('cylinder')
            if old is None or float(old.get('radius'))!=radius:raise ValueError('Expected original cylinder radius')
            geom.remove(old);ET.SubElement(geom,'sphere',radius=str(radius))
        tensor=link.find('./inertial/inertia')
        for k in ('ixx','iyy','izz'):tensor.set(k,str(.4*mass*radius**2))
        origin=joint.find('origin');xyz=[float(x) for x in origin.get('xyz').split()]
        xyz[1]=.40 if xyz[1]>0 else -.40;origin.set('xyz',' '.join(map(str,xyz)));centers.append(xyz)
    if sum(float(l.find('./inertial/mass').get('value')) for l in root.findall('link'))!=40:raise ValueError('Cart mass changed')
    tree.write(path,encoding='utf-8',xml_declaration=True)
    return dict(variant='spherical_tread_passive_wheels',total_mass_kg=40.,wheel_radius_m=radius,wheel_mass_kg=1.,
        wheel_centers_m=centers,wheel_inertia_diagonal_kgm2=[.4*radius**2]*3,
        overall_width_m=1.08,original_cylinder_cart=False,
        note='Round tire cross-section; widened wheel centers prevent body overlap. No drive, reduced gravity, or robot model change.')
