'''
Clusters the given images in groups in order to give representation
'''
import ImageCluster


class ImageGrouping:
  MIN_CLUSTER_STRENGTH = 1.0
  MIN_CLUSTER_SIZE = 2
  MAX_CLUSTER_SIZE = 5

  def split_images(self):
    pass

  def find_representative_image(self, group):
    pass

  def find_image_to_image_distance(self):
    pass

  def prepare_image_cluster(self) -> ImageCluster:
    pass
