xml_start = '''<!DOCTYPE mujoco [
    <!ENTITY size "0.01 0.01 0.01">
    <!ENTITY spheresize ".001 .001 .001">
    <!ENTITY posright "0 .03 0">
    <!ENTITY posup "-.03 0 0">
    <!ENTITY posleft "0 -0.03 0">
    <!ENTITY friction "0">
    <!ENTITY ballfriction ".3 0.005 0.0001">
    <!ENTITY spherefriction "10 0.005 0.0001">
    <!ENTITY density "5">
]>
<mujoco model="composite_object">
  <asset>
    <mesh file="meshes/cube_indent.stl" name="sphere_mesh" scale=".001 .001 .001"/>
  </asset>
  <worldbody>
    <body>
        <body name="object" pos="&posleft;">
            <geom pos="0 0 0" size="&size;" type="box" group="0" friction="10 .005 0.0001" solimp="0.998 0.998 0.000" solref="0.001 1"  condim="4" density="50"/>
'''

xml_end = '''
        </body>
    </body>
  </worldbody>
</mujoco>'''


if __name__=="__main__":
    num_spheres = 5
    file_path = f'./box_{num_spheres}_spheres.xml'
    x_size = 0.02
    z_size = 0.02
    y_size = 0.01
    with open(file_path, 'w') as file:
        file.write(xml_start)
        for i in range(num_spheres):
            for j in range(num_spheres):
                file.write(f'''
                <geom pos="{-1* y_size} {x_size - ((2*x_size)/(num_spheres - 1) * i)} {z_size - ((2*z_size)/(num_spheres - 1) * j)}" size="&spheresize;" type="sphere" group="0" friction="&spherefriction;" solimp="0.998 0.998 0.000" solref="0.001 1"  condim="4" density="5" rgba="1 0 0 1"/>
                <geom pos="{y_size} {x_size - ((2*x_size)/(num_spheres - 1) * i)} {z_size - ((2*z_size)/(num_spheres - 1) * j)}" size="&spheresize;" type="sphere" group="0" friction="&spherefriction;" solimp="0.998 0.998 0.000" solref="0.001 1"  condim="4" density="5" rgba="1 0 0 1"/>
                ''')
        file.write(xml_end)
