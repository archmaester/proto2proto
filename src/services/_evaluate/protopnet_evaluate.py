import os
import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.mgr import manager
from src.utils.dirs import create_dirs
import torch.nn.functional as F
from lib import init_proto_model
from lib.protopnet.losses import cluster_sep_loss_fn, l1_loss_fn
from lib.protopnet.optimizer import get_optimizer, last_only, warm_only, joint
from lib.protopnet import push, save, preprocess
import ray
from lib.utils import evaluate
from src.utils import _common
from scipy.optimize import linear_sum_assignment 
from sklearn.metrics import jaccard_score


class Service(object):

    def __init__(self, dataset_loader):

        self.manager = manager
        self.mgpus = self.manager.common.mgpus
        self.dataset_loader = dataset_loader

        num_cpus = os.cpu_count()
        ray.init(num_cpus=num_cpus)

        self.teacher_model, _, _ = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.Teacherbackbone)

        self.student_kd_model, _, _ = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.StudentKDbackbone)

        self.student_baseline_model, _, _ = init_proto_model(
            manager, dataset_loader.classes, manager.settingsConfig.StudentBaselinebackbone)

    def __call__(self):

        self.teacher_model.eval()
        self.student_kd_model.eval()
        self.student_baseline_model.eval()

        data_iter = iter(self.dataset_loader.test_loader)
        assert manager.common.testBatchSize == 1

        teacher_data = []
        stu_kd_data = []
        stu_baseline_data = []

        for ii, (xs, ys) in enumerate(tqdm(data_iter)):

            with torch.no_grad():
                ys = ys.cuda()
                xs = xs.cuda()

                teacher_distances, _, _ = self.teacher_model.module.prototype_distances(xs)
                stu_kd_distances, _, _ = self.student_kd_model.module.prototype_distances(xs)
                stu_baseline_distances, _, _ = self.student_baseline_model.module.prototype_distances(xs)

                teacher_scores, teacher_indices = F.max_pool2d(-teacher_distances,
                                                  kernel_size=(teacher_distances.size()[2],
                                                               teacher_distances.size()[3]),
                                                  return_indices=True)
                stu_kd_scores, stu_kd_indices = F.max_pool2d(-stu_kd_distances,
                                                 kernel_size=(stu_kd_distances.size()[2],
                                                              stu_kd_distances.size()[3]),
                                                 return_indices=True)
                stu_baseline_scores, stu_baseline_indices = F.max_pool2d(-stu_baseline_distances,
                                                       kernel_size=(stu_baseline_distances.size()[2],
                                                                    stu_baseline_distances.size()[3]),
                                                       return_indices=True)

                teacher_indices = teacher_indices.view(self.teacher_model.module.num_prototypes)
                teacher_scores = teacher_scores.view(self.teacher_model.module.num_prototypes)
                teachers_activations = self.teacher_model.module.distance_2_similarity(-teacher_scores)
                teacher_scores = teacher_scores.detach().cpu().numpy().tolist()
                teacher_indices = teacher_indices.detach().cpu().numpy().tolist()
                teachers_activations = teachers_activations.detach().cpu().numpy().tolist()

                stu_kd_indices = stu_kd_indices.view(self.student_kd_model.module.num_prototypes)
                stu_kd_scores = stu_kd_scores.view(self.student_kd_model.module.num_prototypes)
                stu_kd_activations = self.student_kd_model.module.distance_2_similarity(-stu_kd_scores)
                stu_kd_scores = stu_kd_scores.detach().cpu().numpy().tolist()
                stu_kd_indices = stu_kd_indices.detach().cpu().numpy().tolist()
                stu_kd_activations = stu_kd_activations.detach().cpu().numpy().tolist()

                stu_baseline_indices = stu_baseline_indices.view(self.student_baseline_model.module.num_prototypes)
                stu_baseline_scores = stu_baseline_scores.view(self.student_baseline_model.module.num_prototypes)
                stu_baseline_activations = self.student_baseline_model.module.distance_2_similarity(-stu_baseline_scores)
                stu_baseline_scores = stu_baseline_scores.detach().cpu().numpy().tolist()
                stu_baseline_indices = stu_baseline_indices.detach().cpu().numpy().tolist()
                stu_baseline_activations = stu_baseline_activations.detach().cpu().numpy().tolist()

                teacher_data.append([teacher_indices, teacher_scores])
                stu_kd_data.append([stu_kd_indices, stu_kd_scores])
                stu_baseline_data.append([stu_baseline_indices, stu_baseline_scores])

        self.aap_teacher = []
        self.aap_baseline = []
        self.aap_kd = []

        self.ajs_baseline = []
        self.ajs_kd = []

        distance_thresholds = [0.01, 0.1, 0.2, 0.45, 1.0, 3.0, 5.0, None]
        for dist_th in distance_thresholds:
            self.evaluate(dist_th, teacher_data, stu_kd_data, stu_baseline_data, calc_pm=False)

        plot_aap(self.aap_teacher , self.aap_baseline, self.aap_kd, distance_thresholds, self.manager.base_dir)
        plot_ajs(self.ajs_baseline, self.ajs_kd, distance_thresholds, self.manager.base_dir)

        print("Distance thresholds", distance_thresholds)
        print("AAPs(teacher): ", self.aap_teacher)
        print("AAPs(baseline): ", self.aap_baseline)
        print("AAPs(kd): ", self.aap_kd, "\n\n")
        print("AJS(baseline): ", self.ajs_baseline)
        print("AJS(kd): ", self.ajs_kd)

        self.evaluate(None, teacher_data, stu_kd_data, stu_baseline_data, calc_pm=True)

    def evaluate(self, dist_th, teacher_data, stu_kd_data, stu_baseline_data, calc_pm=True):

        num_test_images = len(self.dataset_loader.test_loader)
        teacher_prototypes = [[[],[]] for ii in range(self.teacher_model.module.num_prototypes)]
        student_kd_prototypes = [[[],[]] for ii in range(self.student_kd_model.module.num_prototypes)]
        student_baseline_prototypes = [[[],[]] for ii in range(self.student_baseline_model.module.num_prototypes)]

        count_tchr = 0
        count_stu_kd = 0
        count_stu_baseline = 0

        iou_tchr = 0.0
        iou_stu_kd = 0.0
        iou_stu_baseline = 0.0

        for ii in tqdm(range(len(teacher_data))):

            if dist_th is None:
                pruned_teacher_indices = teacher_data[ii][0]
                pruned_stu_kd_indices = stu_kd_data[ii][0]
                pruned_stu_baseline_indices = stu_baseline_data[ii][0]
            else:
                pruned_teacher_indices = []
                for jj, score in enumerate(teacher_data[ii][1]):
                    if abs(-score) <= dist_th:
                        pruned_teacher_indices.append(teacher_data[ii][0][jj])

                pruned_stu_kd_indices = []
                for jj, score in enumerate(stu_kd_data[ii][1]):
                    if abs(-score) <= dist_th:
                        pruned_stu_kd_indices.append(stu_kd_data[ii][0][jj])

                pruned_stu_baseline_indices = []
                for jj, score in enumerate(stu_baseline_data[ii][1]):
                    if abs(-score) <= dist_th:
                        pruned_stu_baseline_indices.append(stu_baseline_data[ii][0][jj])

            count_tchr += len(set(pruned_teacher_indices))
            count_stu_kd += len(set(pruned_stu_kd_indices))
            count_stu_baseline += len(set(pruned_stu_baseline_indices))

            iou_stu_kd += jaccard_similarity_basic(pruned_teacher_indices, pruned_stu_kd_indices)
            iou_stu_baseline += jaccard_similarity_basic(pruned_teacher_indices, pruned_stu_baseline_indices)

            for jj in range(len(teacher_prototypes)):
                if jj <= len(pruned_teacher_indices) - 1:
                    name = "%04d" % ii + "%02d" % pruned_teacher_indices[jj]
                    teacher_prototypes[jj][0].append(name)
                    teacher_prototypes[jj][1].append(None)

            for jj in range(len(student_kd_prototypes)):
                if jj <= len(pruned_stu_kd_indices) - 1:
                    name = "%04d" % ii + "%02d" % pruned_stu_kd_indices[jj]
                    student_kd_prototypes[jj][0].append(name)
                    student_kd_prototypes[jj][1].append(None)

            for jj in range(len(student_baseline_prototypes)):
                if jj <= len(pruned_stu_baseline_indices) - 1:
                    name = "%04d" % ii + "%02d" % pruned_stu_baseline_indices[jj]
                    student_baseline_prototypes[jj][0].append(name)
                    student_baseline_prototypes[jj][1].append(None)

        if not calc_pm:
            self.aap_teacher.append(count_tchr / num_test_images)
            self.aap_baseline.append(count_stu_baseline / num_test_images)
            self.aap_kd.append(count_stu_kd / num_test_images)

            self.ajs_baseline.append(iou_stu_baseline / num_test_images)
            self.ajs_kd.append(iou_stu_kd / num_test_images)
            return

        # Teacher, Student-kd IoU
        mm = self.teacher_model.module.num_prototypes
        nn = self.student_kd_model.module.num_prototypes

        tchr_proto_id = ray.put(teacher_prototypes)
        stu_kd_proto_id = ray.put(student_kd_prototypes)
        stu_baseline_proto_id = ray.put(student_baseline_prototypes)
        max_union_list = [ii for ii in range(int(0.1*num_test_images), num_test_images, int(0.1*num_test_images))]

        cost_kd_list = []
        for max_union in max_union_list:

            iou_matrix = np.zeros((mm,nn))
            obj_ids = []
            for ii in tqdm(range(mm)):

                obj_id = jaccard_row.remote(ii, tchr_proto_id, stu_kd_proto_id, max_union)
                obj_ids.append(obj_id)

                if ii % 30 == 0 or ii == mm - 1:
                    results = ray.get(obj_ids)
                    for kk in range(len(obj_ids)):
                        index, sim = results[kk]
                        iou_matrix[index] = sim
                    obj_ids = []

            assert len(obj_ids) == 0

            iou_distance_matrix = 1.0 - iou_matrix
            r_ts , c_stu_kd = linear_sum_assignment(iou_distance_matrix)
            cost_kd = iou_distance_matrix[r_ts, c_stu_kd].sum() / len(r_ts)
            cost_kd_list.append(cost_kd)

        avg_cost_kd = sum(cost_kd_list) / len(cost_kd_list)
        print("Average Similarity between prototypes(KD)", 1.0 - avg_cost_kd)

        # Teacher, Student-baseline IoU
        cost_kd_list = []
        for max_union in max_union_list:

            mm = self.teacher_model.module.num_prototypes
            nn = self.student_baseline_model.module.num_prototypes
            iou_matrix = np.zeros((mm,nn))

            obj_ids = []
            for ii in tqdm(range(mm)):

                obj_id = jaccard_row.remote(ii, tchr_proto_id, stu_baseline_proto_id, max_union)
                obj_ids.append(obj_id)

                if ii % 30 == 0 or ii == mm - 1:
                    results = ray.get(obj_ids)
                    for kk in range(len(obj_ids)):
                        index, sim = results[kk]
                        iou_matrix[index] = sim
                    obj_ids = []

            assert len(obj_ids) == 0

            iou_distance_matrix = 1.0 - iou_matrix
            r_ts , c_stu_baseline = linear_sum_assignment(iou_distance_matrix)
            cost_baseline = iou_distance_matrix[r_ts, c_stu_baseline].sum() / len(r_ts)
            cost_kd_list.append(cost_baseline)

        avg_cost_kd = sum(cost_kd_list) / len(cost_kd_list)
        print("Average Similarity between prototypes(Baseline)", 1.0 - avg_cost_kd)

        ray.get(tchr_proto_id)
        ray.get(stu_kd_proto_id)
        ray.get(stu_baseline_proto_id)

        return

def plot_aap(tchr, baseline, kd, dts, save_dir):

    plt.figure(0)
    #X-Y axis
    x = range(1, len(dts)+1)
    plt.xlabel('Distance Threshold')
    plt.xticks(x, dts)
    plt.ylabel('Average Active Patches')

    # Plot a simple line chart
    plt.plot(x, tchr, 'b', label='Teacher')
    plt.plot(x, baseline, 'r', label='Student(Baseline)')
    plt.plot(x, kd, 'yellow', label='Student(Ours)')

    plt.legend()
    plt.show()
    plt.savefig(os.path.join(save_dir, 'aap.png'))

def plot_ajs(baseline, kd, dts, save_dir):

    plt.figure(1)
    #X-Y axis
    tchr = [1.0]*len(dts)
    x = range(1, len(dts)+1)
    plt.xlabel('Distance Threshold')
    plt.xticks(x, dts)
    plt.ylabel('Average Jaccard similarity with Teacher')

    # Plot a simple line chart
    plt.plot(x, tchr, 'b', label='Teacher')
    plt.plot(x, baseline, 'r', label='Student(Baseline)')
    plt.plot(x, kd, 'yellow', label='Student(Ours)')

    plt.legend()
    plt.show()
    plt.savefig(os.path.join(save_dir, 'ajs.png'))

@ray.remote
def jaccard_row(ii, tchr_prototypes, stu_prototypes, max_union):

    proto_row = np.zeros(len(stu_prototypes))
    for jj in range(len(stu_prototypes)):
        proto_row[jj] = jaccard_similarity(tchr_prototypes[ii], stu_prototypes[jj], max_union=max_union)

    return ii, proto_row


def jaccard_similarity_basic(list1, list2):

    s1 = set(list1)
    s2 = set(list2)

    intersect = len(s1.intersection(s2))
    union = (len(s1) + len(s2)) - intersect

    if union == 0:
        return 0.0

    sim = float(intersect/union)

    return sim


def jaccard_similarity(list1, list2, max_union=100000.0):

    s1 = set(list1[0])
    s2 = set(list2[0])

    intersect = len(s1.intersection(s2))
    union = (len(s1) + len(s2)) - intersect

    if union == 0:
        return 0.0
    elif intersect >= max_union:
        return 1.0
    else:
        sim = float(intersect/min(union, max_union))

    return sim


def jaccard_similarity_modi(list1, list2, max_union=100000.0):

    intersect = 0.0
    for ii, ele in enumerate(list1[0]):
        if ele in list2[0]:
            intersect += list1[1][ii]

    sum_l1 = sum(list1[1])
    sum_l2 = sum(list2[1])

    union = sum_l1 + sum_l2 - intersect

    if union == 0:
        return 0.0

    if intersect >= max_union:
        return 1.0
    else:
        sim = float(intersect/min(union, max_union))

    return sim
