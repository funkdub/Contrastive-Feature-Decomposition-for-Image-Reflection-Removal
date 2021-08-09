import datasyn.reflect_dataset as datasets
from PIL import Image
import numpy as np


def main():
	datadir = '../data/PLNet'
	dataset = datasets.CEILDataset(
                            datadir, fns=None,
                            size=None, enable_transforms=True, 
                            low_sigma=2, high_sigma=5,
                            low_gamma=1.3, high_gamma=1.3)

	# print(len(dataset))
	m_dir = './train/input/'
	b_dir = './train/target'
	for i, item in enumerate(dataset):
		m_tensor = item[0]
		b_tensor = item[1]

		file_name = item[3]
		m_numpy = m_tensor.float().numpy()
		b_numpy = b_tensor.float().numpy()
		m_numpy = np.clip(m_numpy, 0, 1)
		b_numpy = np.clip(b_numpy, 0, 1)

		if m_numpy.shape[0] == 1:
			m_numpy = np.tile(m_numpy, (3, 1, 1))
		if b_numpy.shape[0] == 1:
			b_numpy = np.tile(b_numpy, (3, 1, 1))

		m_numpy = (np.transpose(m_numpy, (1, 2, 0))) * 255.0
		b_numpy = (np.transpose(b_numpy, (1, 2, 0))) * 255.0
		m_numpy = m_numpy.astype(np.uint8)
		b_numpy = b_numpy.astype(np.uint8)
		print(file_name)
		Image.fromarray(m_numpy).save(m_dir + str(file_name))
		Image.fromarray(b_numpy).save(b_dir + str(file_name))


if __name__ == '__main__':
    main()


