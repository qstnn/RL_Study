"""无仿真依赖的被动车URDF与牵引杆力学。"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import torch

RADIUS=.14
BODY_MASS=36.
WHEEL_MASS=1.
ROBOT_HITCH=(-.38,0.,.02)
CART_HITCH=(.35,0.,.15)


def write_cart_urdf(destination):
    """运行目录内生成40kg被动车；箱体36kg、四轮各1kg，没有驱动或隐藏牵引推力。"""
    root=ET.Element("robot",name="r1_passive_cart")
    base=ET.SubElement(root,"link",name="cart_base")
    def inertial(link,mass,diagonal):
        element=ET.SubElement(link,"inertial")
        ET.SubElement(element,"origin",xyz="0 0 0",rpy="0 0 0")
        ET.SubElement(element,"mass",value=str(mass))
        ET.SubElement(element,"inertia",ixx=str(diagonal[0]),iyy=str(diagonal[1]),izz=str(diagonal[2]),ixy="0",ixz="0",iyz="0")
    inertial(base,BODY_MASS,[BODY_MASS*(.5**2+.16**2)/12,BODY_MASS*(.7**2+.16**2)/12,BODY_MASS*(.7**2+.5**2)/12])
    for kind in ("visual","collision"):
        geom=ET.SubElement(ET.SubElement(base,kind),"geometry")
        ET.SubElement(geom,"box",size="0.7 0.5 0.16")
    for index,(x,y) in enumerate(((.25,.31),(.25,-.31),(-.25,.31),(-.25,-.31))):
        name=f"cart_wheel_{index}"
        link=ET.SubElement(root,"link",name=name)
        transverse=WHEEL_MASS*(3*RADIUS**2+.06**2)/12
        inertial(link,WHEEL_MASS,[transverse,WHEEL_MASS*RADIUS**2/2,transverse])
        for kind in ("visual","collision"):
            item=ET.SubElement(link,kind)
            ET.SubElement(item,"origin",xyz="0 0 0",rpy=f"{math.pi/2} 0 0")
            ET.SubElement(ET.SubElement(item,"geometry"),"cylinder",radius=str(RADIUS),length=".06")
        joint=ET.SubElement(root,"joint",name=name+"_joint",type="continuous")
        ET.SubElement(joint,"parent",link="cart_base")
        ET.SubElement(joint,"child",link=name)
        ET.SubElement(joint,"origin",xyz=f"{x} {y} -.16",rpy="0 0 0")
        ET.SubElement(joint,"axis",xyz="0 1 0")
        ET.SubElement(joint,"limit",effort="0",velocity="200")
        ET.SubElement(joint,"dynamics",damping="0",friction="0")
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(destination,encoding="utf-8",xml_declaration=True)


def drawbar_force(robot_point,cart_point,robot_velocity,cart_velocity,active,rest=1.,stiffness=2000.,damping=150.):
    """对两端施加等大反向牵引杆力，含拉伸与压缩；只使用实际状态，不读速度指令。"""
    delta=robot_point-cart_point
    length=delta.norm(dim=-1).clamp_min(1e-6)
    direction=delta/length[:,None]
    extension_rate=((robot_velocity-cart_velocity)*direction).sum(-1)
    tension=stiffness*(length-rest)+damping*extension_rate
    # 力上限是数值保护，触及上限单独计数且评估失败，不能冒充完整牵引结果。
    force=tension.clamp(-1500.,1500.)[:,None]*direction*active[:,None]
    return force,length,tension*active


