import gzip
import logging
import os
import shutil
import tarfile
import zipfile


def make_dir(file_path):
    """
    package tar.gz file
    :param file_path: target file path
    :param process_id: current start edge id
    :return: bool
    """
    try:
        os.makedirs(file_path)
        return True
    except Exception as e:
        logging.exception(e)
        return False


def check_is_download(path):
    if os.path.isdir(path):
        logging.info(f'Edge Data exist.')
        return True
    else:
        return False


def file_extract(root: str, file_path: str):
    if file_path.endswith('.zip'):
        return un_zip(file_path)
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        return un_tar(root, file_path)
    else:
        return None


def un_zip(file_name):
    """unzip zip file"""
    dest_dir = file_name
    try:
        with zipfile.ZipFile(file_name) as zip_file:
            if not os.path.isdir(dest_dir):
                os.mkdir(file_name)
            zip_file.extractall(path=dest_dir)
        return dest_dir
    except Exception as e:
        shutil.rmtree(dest_dir)
        logging.exception(e)
        return None


def un_gz(file_name):
    """un_gz zip file"""
    # get file name without suffix
    f_name = file_name.replace(".gz", "")
    try:
        with gzip.GzipFile(file_name) as g_file:
            with open(f_name, "wb") as dest_file:
                dest_file.write(g_file.read())
        return f_name
    except Exception as e:
        shutil.rmtree(f_name)
        logging.exception(e)
        return None


def un_tar(root, file_name):
    """ untar zip file"""
    dest_dir = os.path.join(root, 'cifar-10-batches-py')
    try:
        with tarfile.open(file_name) as tar:
            if not os.path.isdir(dest_dir):
                os.mkdir(dest_dir)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=dest_dir)
            logging.info("untar zip file finished")
        return dest_dir
    except Exception as e:
        shutil.rmtree(dest_dir)
        logging.exception(e)
        return None


def move_file(edge_id, root):
    target_src = os.path.join(root, 'cifar-10-batches-py', 'device_%s' % edge_id)
    dirs = os.listdir(target_src)
    for file_name in dirs:
        shutil.move(os.path.join(target_src, file_name), os.path.join(root, 'cifar-10-batches-py'))
    logging.info("Move file finished")
    return None