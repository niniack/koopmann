"""
Twelve broad groups of ImageNet's 1,000 classes, built by hand for the review (no class tree is on this machine).

Animals: ImageNet's class order follows WordNet, so each kind of animal is a run of class indices.
Objects (classes 398 to 999, in alphabetical order): each class was put in a group by reading its name.
Whatever is not listed lands in "Household and other objects". `class_groups()` gives the group of every class.
"""

import torch

runs = lambda *pairs: [i for a, b in pairs for i in range(a, b + 1)]

GROUPS = {
    "Dogs": runs((151, 268)),
    "Other mammals": runs((101, 106), (147, 150), (269, 299), (330, 388)),
    "Birds": runs((7, 24), (80, 100), (127, 146)),
    "Fish, reptiles, amphibians": runs((0, 6), (25, 68), (389, 397)),
    "Invertebrates": runs((69, 79), (107, 126), (300, 329)),
    "Vehicles": [403, 404, 405, 407, 408, 417, 436, 444, 450, 466, 468, 472, 484, 510, 511, 537, 547, 554, 555, 561, 565, 569, 573, 575, 576, 586, 595, 603, 609, 612,
                 625, 627, 628, 654, 656, 661, 665, 670, 671, 675, 690, 705, 717, 724, 734, 751, 757, 779, 780, 802, 803, 812, 814, 817, 820, 829, 833, 847, 864, 866,
                 867, 870, 871, 874, 880, 895, 914],
    "Clothing": [399, 400, 411, 433, 439, 443, 445, 451, 452, 457, 459, 461, 465, 474, 490, 496, 501, 502, 514, 515, 518, 524, 529, 552, 560, 568, 570, 578, 601, 608,
                 610, 614, 615, 617, 630, 638, 639, 643, 652, 655, 658, 667, 679, 689, 697, 715, 735, 770, 774, 775, 793, 796, 806, 808, 824, 834, 836, 837, 841, 842,
                 869, 887, 903, 906],
    "Buildings and scenes": [406, 410, 415, 424, 425, 437, 442, 449, 454, 460, 467, 483, 489, 497, 498, 500, 509, 525, 536, 538, 562, 580, 582, 624, 634, 646, 649, 660,
                             663, 668, 672, 682, 698, 706, 716, 718, 727, 743, 762, 788, 819, 821, 825, 832, 839, 853, 858, 860, 863, 865, 873, 884, 888, 900, 912, 915,
                             970, 972, 973, 974, 975, 976, 977, 978, 979, 980],
    "Food, plants, fungi": runs((924, 967), (969, 969), (984, 998)),
    "Musical instruments": [401, 402, 420, 432, 486, 494, 513, 541, 542, 546, 558, 566, 577, 579, 593, 594, 641, 642, 683, 684, 687, 699, 714, 776, 822, 875, 881, 889, 902],
    "Devices and machines": [409, 426, 447, 480, 481, 482, 485, 487, 491, 505, 508, 517, 527, 528, 530, 531, 534, 540, 545, 548, 550, 571, 589, 590, 592, 598, 605, 606,
                             613, 620, 621, 632, 635, 650, 651, 662, 664, 673, 681, 685, 688, 704, 707, 713, 732, 740, 742, 745, 753, 754, 755, 759, 760, 761, 766, 778,
                             782, 786, 800, 807, 810, 811, 818, 826, 827, 844, 848, 851, 856, 859, 878, 882, 886, 891, 892, 897],
}  # fmt: skip
OTHER = "Household and other objects"
NAMES = [*GROUPS, OTHER]
ANIMALS, PICTURE_GROUPS = NAMES[:5], ["Other mammals", "Dogs", "Vehicles", "Clothing", "Musical instruments", "Devices and machines", OTHER]


def class_groups():
    """The group of every class, as an index into NAMES. shape: [1000]"""

    group = torch.full((1000,), len(GROUPS))
    for index, classes in enumerate(GROUPS.values()):
        assert (group[classes] == len(GROUPS)).all(), "a class is in two groups"
        group[classes] = index
    return group


if __name__ == "__main__":
    import open_clip

    group = class_groups()
    for index, name in enumerate(NAMES):
        members = (group == index).nonzero()[:, 0].tolist()
        print(f"\n{name} ({len(members)} classes): " + ", ".join(open_clip.IMAGENET_CLASSNAMES[c] for c in members))
