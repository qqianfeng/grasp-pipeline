import urllib2
import wget
import tarfile
import os
import subprocess

GDL_MODEL_PATH = '/home/vm'

#location for files that are ready for use with gazebo and graspit
PROCESSED_OUTPUT_DIR = GDL_MODEL_PATH + "/big_bird_models_processed"

#location to download the raw files into
RAW_DOWNLOAD_DIR = GDL_MODEL_PATH + '/raw_big_bird_models'
#temp file used since each model is downloaded individually
TAR_FILE_NAME = 'processed.tgz'


def run_subprocess(cmd_string):
    print "running: " + cmd_string

    process = subprocess.Popen(cmd_string.split(), stdout=subprocess.PIPE)
    output = process.communicate()[0]

    print output


def download_bigbird_models():

    if not os.path.exists(RAW_DOWNLOAD_DIR):
        os.mkdir(RAW_DOWNLOAD_DIR)

    # url = "http://rll.berkeley.edu/bigbird/aliases/772151f9ac/"
    url = "https://drive.google.com/drive/folders/1sX9YXYqsr47Dee2zicEWleHe10KtklZx"
    req = urllib2.Request(url)
    res = urllib2.urlopen(req)
    html_split = res.read().split('application/vnd.google-apps.folder')
    #print(html_split)

    model_names = []
    for txt in html_split:
        model_name = txt.split(',')[-2]
        if model_name != 'null' and model_name != '"export"':
            model_name = model_name[1:-1]
            model_names.append(model_name)
            print(model_name),

    for model_name in model_names:
        print ""
        print model_name
        if not os.path.exists(RAW_DOWNLOAD_DIR + '/' + model_name):
            if os.path.exists(os.getcwd() + '/' + TAR_FILE_NAME):
                os.remove(os.getcwd() + '/' + TAR_FILE_NAME)

            download_url = "http://rll.berkeley.edu/bigbird/aliases/772151f9ac/export/" + model_name + "/" + TAR_FILE_NAME
            wget.download(download_url)
            t = tarfile.open(os.getcwd() + '/' + TAR_FILE_NAME, 'r')
            t.extractall(RAW_DOWNLOAD_DIR)


def prep_meshes_for_gazebo_and_graspit():

    if os.path.exists(PROCESSED_OUTPUT_DIR):
        for subdir in os.listdir(PROCESSED_OUTPUT_DIR):
            for filename in os.listdir(PROCESSED_OUTPUT_DIR + "/" + subdir):
                os.remove(PROCESSED_OUTPUT_DIR + "/" + subdir + "/" + filename)
            os.rmdir(PROCESSED_OUTPUT_DIR + "/" + subdir)
        os.rmdir(PROCESSED_OUTPUT_DIR)

    os.mkdir(PROCESSED_OUTPUT_DIR)
    models = os.listdir(RAW_DOWNLOAD_DIR)

    for model_name in models:

        raw_model_dir = RAW_DOWNLOAD_DIR + "/" + model_name + "/"
        processed_model_dir = PROCESSED_OUTPUT_DIR + "/" + model_name + "/"

        os.mkdir(processed_model_dir)

        ################################
        # write the .iv file
        ################################
        cmd = ""
        cmd += "rosrun ivcon ivcon "
        cmd += raw_model_dir + "textured_meshes/optimized_tsdf_texture_mapped_mesh.obj "
        cmd += processed_model_dir + model_name + ".iv"

        run_subprocess(cmd)

        ################################
        # write the .xml file
        ################################
        model_xml = ""
        model_xml += '<?xml version="1.0" ?>\n'
        model_xml += '  <root>\n'
        model_xml += '     <geometryFile type="Inventor">' + model_name + '.iv' + '</geometryFile>\n'
        model_xml += '  </root>'

        f = open(processed_model_dir + model_name + ".xml", 'w')
        f.write(model_xml)
        f.close()

        ################################
        # copy in the texture file
        ################################
        cmd = ""
        cmd += 'cp '
        cmd += raw_model_dir + "textured_meshes/optimized_tsdf_texture_mapped_mesh.png "
        cmd += processed_model_dir

        run_subprocess(cmd)

        ################################
        # copy in the obj file
        ################################
        cmd = ""
        cmd += 'cp '
        cmd += raw_model_dir + "textured_meshes/optimized_tsdf_texture_mapped_mesh.obj "
        cmd += processed_model_dir

        run_subprocess(cmd)

        ################################
        # copy in the mtl file
        ################################
        cmd = ""
        cmd += 'cp '
        cmd += raw_model_dir + "textured_meshes/optimized_tsdf_texture_mapped_mesh.mtl "
        cmd += processed_model_dir

        run_subprocess(cmd)

        ################################
        # write the model.config
        ################################
        model_config = ""
        model_config += '<?xml version="1.0"?>\n'
        model_config += '  <model>\n'
        model_config += '    <name>' + model_name + '</name>\n'
        model_config += '    <version>1.0</version>\n'
        model_config += '    <sdf version="1.2">model-1_2.sdf</sdf>\n'
        model_config += '    <sdf version="1.3">model-1_3.sdf</sdf>\n'
        model_config += '    <sdf version="1.4">model-1_4.sdf</sdf>\n'
        model_config += '    <sdf version="1.5">model.sdf</sdf>\n'
        model_config += '  </model>'

        f = open(processed_model_dir + "model.config", 'w')
        f.write(model_config)
        f.close()

        ################################
        # write the model.sdf
        ################################
        model_sdf = ""
        model_sdf += '<?xml version="1.0" ?>\n'
        model_sdf += '<sdf version="1.5">\n'
        model_sdf += '  <model name="' + model_name + '">\n'
        model_sdf += '    <link name="link">\n'
        model_sdf += '      <inertial>\n'
        model_sdf += '        <pose>-10.01 -0.012 0.15 0 0 0</pose>\n'
        model_sdf += '        <mass>0.390</mass>\n'
        model_sdf += '       <inertia>\n'
        model_sdf += '          <ixx>0.00058</ixx>\n'
        model_sdf += '         <ixy>0</ixy>\n'
        model_sdf += '          <ixz>0</ixz>\n'
        model_sdf += '          <iyy>0.00058</iyy>\n'
        model_sdf += '          <iyz>0</iyz>\n'
        model_sdf += '          <izz>0.00019</izz>\n'
        model_sdf += '        </inertia>\n'
        model_sdf += '      </inertial>\n'
        model_sdf += '      <collision name="collision">\n'
        model_sdf += '        <pose>0 0 0 0 0 0</pose>\n'
        model_sdf += '        <geometry>\n'
        model_sdf += '          <mesh>\n'
        model_sdf += '            <uri>model://' + model_name + '/optimized_tsdf_texture_mapped_mesh.dae</uri>\n'
        model_sdf += '          </mesh>\n'
        model_sdf += '        </geometry>\n'
        model_sdf += '        <surface>\n'
        model_sdf += '          <friction>\n'
        model_sdf += '            <ode>\n'
        model_sdf += '              <mu>1.0</mu>\n'
        model_sdf += '              <mu2>1.0</mu2>\n'
        model_sdf += '            </ode>\n'
        model_sdf += '         </friction>\n'
        model_sdf += '          <contact>\n'
        model_sdf += '            <ode>\n'
        model_sdf += '              <kp>10000000.0</kp>\n'
        model_sdf += '              <kd>1.0</kd>\n'
        model_sdf += '              <min_depth>0.001</min_depth>\n'
        model_sdf += '              <max_vel>0.1</max_vel>\n'
        model_sdf += '            </ode>\n'
        model_sdf += '          </contact>\n'
        model_sdf += '        </surface>\n'
        model_sdf += '      </collision>\n'
        model_sdf += '      <visual name="visual">\n'
        model_sdf += '        <pose>0 0 0 0 0 0</pose>\n'
        model_sdf += '        <geometry>\n'
        model_sdf += '          <mesh>\n'
        model_sdf += '            <uri>model://' + model_name + '/optimized_tsdf_texture_mapped_mesh.dae</uri>\n'
        model_sdf += '          </mesh>\n'
        model_sdf += '        </geometry>\n'
        model_sdf += '      </visual>\n'
        model_sdf += '    </link>\n'
        model_sdf += '  </model>\n'
        model_sdf += '</sdf>\n'

        f = open(processed_model_dir + "model.sdf", 'w')
        f.write(model_sdf)
        f.close()

        f = open(processed_model_dir + "model-1_4.sdf", 'w')
        f.write(model_sdf)
        f.close()


download_bigbird_models()
prep_meshes_for_gazebo_and_graspit()
