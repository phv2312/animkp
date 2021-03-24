joint_labels_dct = {
    'body': 0,
    'nose': 1,
    'larm' : 2,
    'lelbow': 3,
    'lwrist' : 4,
    'rarm': 5,
    'relbow': 6,
    'rwrist': 7,
    'lleg': 8,
    'lknee': 9,
    'lankle': 10,
    'rleg': 11,
    'rknee' : 12,
    'rankle': 13,
    'leye': 14,
    'reye': 15,
    'chin': 16,
    'mouth': 17,
}
joint_labels = list(joint_labels_dct.keys())
joint_pair = [
    ('reye', 'nose'), ('leye', 'nose'), ('nose', 'mouth'), ('mouth', 'chin'),
    ('chin', 'body'),
    ('body', 'larm'), ('body', 'rarm'), ('larm', 'lelbow'), ('rarm', 'relbow'), ('lelbow', 'lwrist'), ('relbow', 'rwrist'),
    ('body', 'lleg'), ('body', 'rleg'), ('lleg', 'lknee'), ('rleg', 'rknee'), ('lknee', 'lankle'), ('rknee', 'rankle')
]

flip_pairs = [[14, 15], [10, 13], [9, 12], [8, 11], [4, 7], [2, 5], [3, 6]]
