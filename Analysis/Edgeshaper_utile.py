# ------------------------------------------------------------
# EdgeSHAPer-based explanation + fragment aggregation utilities
# ------------------------------------------------------------
import os
import math
import time
import textwrap
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import cm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from rdkit import Chem
from rdkit.Chem import Draw

from PIL import Image

### Module implementing EdgeSHAPer algorithm ###
### Author: Andrea Mastropietro © All rights reserved ###

import torch
import torch.nn.functional as F
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw

###EdgeSHAPer as a class###

class Edgeshaper():
    """ EdgeSHAPer class for Shaplay value approximation for edge importance in GNNs
        
            P (float, optional): probablity of an edge to exist in the random graph. Defaults to the original graph density.
            M (int): number of Monte Carlo sampling steps to perform.
            target_class (int, optional): index of the class prediction to explain. Defaults to class 0. None if the model is a regression model.
            deviation (float, optional): deviation gap from sum of Shapley values and predicted output prob. If ```None```, M sampling
                steps are computed, otherwise the procedure stops when the desired approxiamtion ```deviation``` is reached. However, after M steps
                the procedure always terminates.
            log_odds (bool, optional). Default is ```False```. If ```True```, use log odds instead of probabilty as Value for Shaply values approximation.
            seed (float, optional): seed used for the random number generator.
            device (string, optional): states if using ```cuda``` or ```cpu```.
        Returns:
                list: list of Shapley values for the edges computed by EdgeSHAPer. The order of the edges is the same as in ```E```.
    """

    def __init__(self, model, x, edge_index, edge_weight = None, device = "cpu"):
        """ Args:
            model (Torch.NN.Module): Torch GNN model used.
            x (tensor): 2D tensor containing node features of the graph to explain
            edge_index (tensor): edge index of the graph to be explained.
            edge_weight (tensor): weights of the edges.
            device (string, optional): states if using ```cuda``` or ```cpu```.
        """
        super(Edgeshaper, self).__init__()
        # torch.manual_seed(12345)
        self.model = model
        self.model.to(device)
        self.x = x.to(device)
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.device = device


        self.phi_edges = None
        
        self.target_class = None
        self.explained = False

        self.pertinent_positive_set = None
        self.minimal_top_k_set = None
        self.fidelity = None
        self.infidelity = None
        self.trustuworthiness = None

        self.original_pred_prob = None

    def explain(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None):
        """ Compute edge importance using EdgeSHAPer algorithm.
            M (int): number of Monte Carlo sampling steps to perform.
            target_class (int, optional): index of the class prediction to explain. Defaults to class 0. None if the model is a regression model.
            P (float, optional): probablity of an edge to exist in the random graph. Defaults to the original graph density.
            deviation (float, optional): deviation gap from sum of Shapley values and predicted output prob. If ```None```, M sampling
                steps are computed, otherwise the procedure stops when the desired approxiamtion ```deviation``` is reached. However, after M steps
                the procedure always terminates.
            log_odds (bool, optional). Default is ```False```. If ```True```, use log odds instead of probabilty as Value for Shapley values approximation.
            seed (float, optional): seed used for the random number generator.
        Returns:
                list: list of Shapley values for the edges computed by EdgeSHAPer. The order of the edges is the same as in ```self.edge_index```.
        """

        if deviation is not None:
            return self.explain_with_deviation(M = M, target_class = target_class, P = P, deviation = deviation, log_odds = log_odds, seed = seed, device=self.device)

        if target_class is None:
            print("No target class specified. Regression model assumed.")

        E = self.edge_index
        rng = default_rng(seed = seed)
        self.model.eval()
        phi_edges = []

        num_nodes = self.x.shape[0]
        num_edges = E.shape[1]
        
        if P == None:
            max_num_edges = num_nodes*(num_nodes-1)
            graph_density = num_edges/max_num_edges
            P = graph_density

        for j in tqdm(range(num_edges)):
            marginal_contrib = 0
            for i in range(M):
                E_z_mask = rng.binomial(1, P, num_edges)
                E_mask = torch.ones(num_edges)
                pi = torch.randperm(num_edges)

                E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
                E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
                selected_edge_index = np.where(pi == j)[0].item()
                for k in range(num_edges):
                    if k <= selected_edge_index:
                        E_j_plus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

                for k in range(num_edges):
                    if k < selected_edge_index:
                        E_j_minus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_minus_index[pi[k]] = E_z_mask[pi[k]]

                # with edge j (PLUS) --- 그대로 OK
                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(self.device).squeeze()
                E_j_plus = torch.index_select(E, dim=1, index=retained_indices_plus)
                edge_weight_j_plus = None
                if self.edge_weight is not None:
                    edge_weight_j_plus = torch.index_select(self.edge_weight, dim=0, index=retained_indices_plus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                out = self.model(self.x, E_j_plus, batch=batch, edge_weight=edge_weight_j_plus)
                if target_class is not None:
                    V_j_plus = (F.softmax(out, dim=1)[0][target_class]).item() if not log_odds else out[0][target_class].item()
                else:
                    V_j_plus = out.view(-1)[0].item()

                # without edge j (MINUS) --- ✅ 여기서 minus를 호출해야 함
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(self.device).squeeze()
                E_j_minus = torch.index_select(E, dim=1, index=retained_indices_minus)
                edge_weight_j_minus = None
                if self.edge_weight is not None:
                    edge_weight_j_minus = torch.index_select(self.edge_weight, dim=0, index=retained_indices_minus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                out = self.model(self.x, E_j_minus, batch=batch, edge_weight=edge_weight_j_minus)  # ✅
                if target_class is not None:
                    V_j_minus = (F.softmax(out, dim=1)[0][target_class]).item() if not log_odds else out[0][target_class].item()
                else:
                    V_j_minus = out.view(-1)[0].item()


                marginal_contrib += (V_j_plus - V_j_minus)

            phi_edges.append(marginal_contrib/M)

        self.target_class = target_class
        self.explained = True    
        self.phi_edges = phi_edges
        return phi_edges


    def explain_with_deviation(self, M = 100, target_class = 0, P = None, deviation = None, log_odds = False, seed = None, device = "cpu"):
        
        if target_class is None:
            print("No target class specified. Regression model assumed.")
            
        rng = default_rng(seed = seed)
        self.model.eval()
        batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
        E = self.edge_index
        out = self.model(self.x, E, batch=batch, edge_weight=self.edge_weight)
        out_prob_real = F.softmax(out, dim = 1)[0][target_class].item() if target_class is not None else out[0][0].item()

        num_nodes = self.x.shape[0]
        num_edges = E.shape[1]

        phi_edges = []
        phi_edges_current = [0] * num_edges
        
        if P == None:
            max_num_edges = num_nodes*(num_nodes-1)
            graph_density = num_edges/max_num_edges
            P = graph_density

        for i in tqdm(range(M)):
        
            for j in range(num_edges):
                
                
                E_z_mask = rng.binomial(1, P, num_edges)
                E_mask = torch.ones(num_edges)
                pi = torch.randperm(num_edges)

                E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
                E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
                selected_edge_index = np.where(pi == j)[0].item()
                for k in range(num_edges):
                    if k <= selected_edge_index:
                        E_j_plus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

                for k in range(num_edges):
                    if k < selected_edge_index:
                        E_j_minus_index[pi[k]] = E_mask[pi[k]]
                    else:
                        E_j_minus_index[pi[k]] = E_z_mask[pi[k]]


                #we compute marginal contribs
                
                # with edge j
                retained_indices_plus = torch.LongTensor(torch.nonzero(E_j_plus_index).tolist()).to(device).squeeze()
                E_j_plus = torch.index_select(E, dim = 1, index = retained_indices_plus)
                edge_weight_j_plus = None

                if self.edge_weight is not None:
                    edge_weight_j_plus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_plus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                
                out = self.model(self.x, E_j_plus, batch=batch, edge_weight=edge_weight_j_plus)
                out_prob = None

                V_j_plus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_plus = out_prob[0][target_class].item()
                else:
                    
                    out_prob = out
                
                    V_j_plus = out_prob[0][0].item()

                # without edge j
                retained_indices_minus = torch.LongTensor(torch.nonzero(E_j_minus_index).tolist()).to(device).squeeze()
                E_j_minus = torch.index_select(E, dim = 1, index = retained_indices_minus)
                edge_weight_j_minus = None

                if self.edge_weight is not None:
                    edge_weight_j_minus = torch.index_select(self.edge_weight, dim = 0, index = retained_indices_minus)

                batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
                out = self.model(self.x, E_j_minus, batch=batch, edge_weight=edge_weight_j_minus)

                V_j_minus = None
                
                if target_class is not None:
                    if not log_odds:
                        out_prob = F.softmax(out, dim = 1)
                    else:
                        out_prob = out #out prob variable now containts log_odds

                    V_j_minus = out_prob[0][target_class].item()
                else:
                    out_prob = out
                
                    V_j_minus = out_prob[0][0].item()

                phi_edges_current[j] += (V_j_plus - V_j_minus)

            
            phi_edges = [elem / (i+1) for elem in phi_edges_current]
            # print(sum(phi_edges))
            if abs(out_prob_real - sum(phi_edges)) <= deviation:
                break

        self.phi_edges = phi_edges
        self.target_class = target_class
        self.explained = True    
        return phi_edges

    def compute_original_predicted_probability(self):
        self.model.eval()
        batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
        out_log_odds = self.model(self.x, self.edge_index, batch=batch, edge_weight=self.edge_weight)
        out_prob = F.softmax(out_log_odds, dim = 1)
        original_pred_prob = out_prob[0][self.target_class].item()

        self.original_pred_prob = original_pred_prob

    def compute_pertinent_positive_set(self, verbose = False):
        assert(self.explained) #make sure that the explanation has been computed
        
        if self.target_class is None:
            raise Exception("Minimal informative sets are not defined for regression problems.")

        if self.original_pred_prob is None:
            self.compute_original_predicted_probability()

        self.model.eval()
        infidelity = 1 #None it was none, last edit since it remains none if the class does not change, so we put 1
        important_edges_ranking = np.argsort(-np.array(self.phi_edges))
        for i in range(1, important_edges_ranking.shape[0]+1):
            reduced_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[0:i]).to(self.device))
            batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
            out = self.model(self.x, reduced_edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            # print(out_prob)
            predicted_class = torch.argmax(out_prob[0]).item()
            if (predicted_class == self.target_class):
                
                
                pred_prob = out_prob[0][self.target_class].item()
                infidelity = self.original_pred_prob-pred_prob

                if verbose:
                    print("FID- using pertinent positive set: ", infidelity)
                break

        self.pertinent_positive_set = reduced_edge_index
        self.infidelity = infidelity
        return reduced_edge_index, infidelity

    #for legacy with old code, we keep the mispelled function which is now deprecated but calls the correct one
    def compute_pertinent_positivite_set(self, verbose = False):
        print("WARNING: compute_pertinent_positivite_set is now deprecated, use compute_pertinent_positive_set instead.")
        return self.compute_pertinent_positive_set(verbose)

    def compute_minimal_top_k_set(self, verbose = False):
        assert(self.explained) #make sure that the explanation has been computed
        
        if self.target_class is None:
            raise Exception("Minimal informative sets are not defined for regression problems.")
            
        if self.original_pred_prob is None:
            self.compute_original_predicted_probability()

        self.model.eval()
        fidelity = 0 #it was None, last edit since it remains none if the class does not change, so we put 0
        pertinent_set_indices = []
        pertinent_set_edge_index = None
        important_edges_ranking = np.argsort(-np.array(self.phi_edges))
        for i in range(important_edges_ranking.shape[0]):
            index_of_edge_to_remove = important_edges_ranking[i]
            pertinent_set_indices.append(index_of_edge_to_remove)

            reduced_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(important_edges_ranking[i:]).to(self.device))
            
            # all nodes belong to same graph
            batch = torch.zeros(self.x.shape[0], dtype=int, device=self.x.device)
            out = self.model(self.x, reduced_edge_index, batch=batch)
            out_prob = F.softmax(out, dim = 1)
            # print(out_prob)
            predicted_class = torch.argmax(out_prob[0]).item()

            if predicted_class != self.target_class:
                pred_prob = out_prob[0][self.target_class].item()
                fidelity = self.original_pred_prob - pred_prob
                if verbose:
                    print("FID+ using minimal top-k set: ", fidelity)
                break

        pertinent_set_edge_index = torch.index_select(self.edge_index, dim = 1, index = torch.LongTensor(pertinent_set_indices).to(self.device))
        
        self.minimal_top_k_set = pertinent_set_edge_index
        self.fidelity = fidelity

        return pertinent_set_edge_index, fidelity


    def compute_trustworthiness(self, verbose = False):
        ''' 
        Computes the Trustworthiness (TW) of the explanation. The TW metric is defined as an harmonic mean of the fidelity and reverse of the infidelity. 
        '''

        assert(self.explained) #make sure that the explanation has been computed

        assert(self.fidelity is not None) #make sure that the fidelity has been computed
        assert(self.infidelity is not None) #make sure that the infidelity has been computed

        TW = None
        
        if self.fidelity+(1-self.infidelity) == 0:
            TW = 0
        else:
            TW = 2* ((self.fidelity*(1-self.infidelity))/(self.fidelity+(1-self.infidelity)))

        self.trustworthiness = TW

        if verbose:
            print("Trustworthiness: ", self.trustuworthiness)

        return self.trustworthiness

    def visualize_molecule_explanations(self, smiles, save_path=None, pertinent_positive = False, minimal_top_k = False):
        assert(self.explained) #make sure that the explanation has been computed

        img_expl = None
        img_pert_pos = None
        img_min_top_k = None

        edge_index = self.edge_index.to("cpu")

        test_mol = Chem.MolFromSmiles(smiles)
        test_mol = Draw.PrepareMolForDrawing(test_mol)

        num_bonds = len(test_mol.GetBonds())

        rdkit_bonds = {}

        for i in range(num_bonds):
            init_atom = test_mol.GetBondWithIdx(i).GetBeginAtomIdx()
            end_atom = test_mol.GetBondWithIdx(i).GetEndAtomIdx()
            
            rdkit_bonds[(init_atom, end_atom)] = i

        rdkit_bonds_phi = [0]*num_bonds
        for i in range(len(self.phi_edges)):
            phi_value = self.phi_edges[i]
            init_atom = edge_index[0][i].item()
            end_atom = edge_index[1][i].item()
            
            if (init_atom, end_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(init_atom, end_atom)]
                rdkit_bonds_phi[bond_index] += phi_value
            if (end_atom, init_atom) in rdkit_bonds:
                bond_index = rdkit_bonds[(end_atom, init_atom)]
                rdkit_bonds_phi[bond_index] += phi_value

        plt.clf()
        canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi, atom_width=0.2, bond_length=0.5, bond_width=0.5) #TBD: only one direction for edges? bonds weights is wrt rdkit bonds order?
        img_expl = transform2png(canvas.GetDrawingText())

        if save_path is not None:
            img_expl.save(save_path + "/" + "EdgeSHAPer_explanations_heatmap.png", dpi = (300,300))

        if pertinent_positive:
            if self.pertinent_positive_set is None:
                self.compute_pertinent_positivite_set()

            rdkit_bonds_phi_pertinent_set = [0]*num_bonds
            pertinent_set_edge_index = self.pertinent_positive_set
            for i in range(pertinent_set_edge_index.shape[1]):
                
                init_atom = pertinent_set_edge_index[0][i].item()
                end_atom = pertinent_set_edge_index[1][i].item()
                
                
                if (init_atom, end_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(init_atom, end_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]
                if (end_atom, init_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(end_atom, init_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]

            plt.clf()
            canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
            img_pert_pos = transform2png(canvas.GetDrawingText())

            if save_path is not None:

                img_pert_pos.save(save_path + "/" + "EdgeSHAPer_pertinent_positive_set_heatmap.png", dpi = (300,300))

        if minimal_top_k:
            if self.minimal_top_k_set is None:
                self.compute_minimal_top_k_set()

            rdkit_bonds_phi_pertinent_set = [0]*num_bonds
            pertinent_set_edge_index = self.minimal_top_k_set
            for i in range(pertinent_set_edge_index.shape[0]):
                
                init_atom = pertinent_set_edge_index[0][i].item()
                end_atom = pertinent_set_edge_index[1][i].item()
                
                
                if (init_atom, end_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(init_atom, end_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]
                if (end_atom, init_atom) in rdkit_bonds:
                    bond_index = rdkit_bonds[(end_atom, init_atom)]
                    if rdkit_bonds_phi_pertinent_set[bond_index] == 0:
                        rdkit_bonds_phi_pertinent_set[bond_index] += rdkit_bonds_phi[bond_index]

            plt.clf()
            canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi_pertinent_set, atom_width=0.2, bond_length=0.5, bond_width=0.5)
            img_min_top_k = transform2png(canvas.GetDrawingText())

            if save_path is not None:
                img_min_top_k.save(save_path + "/" + "EdgeSHAPer_minimal_top_k_set_heatmap.png", dpi = (300,300))

        plt.clf()
        return img_expl, img_pert_pos, img_min_top_k    


import copy

def batch_edgeshaper(model, batch, M=100, target=0, P=None, seed=42, target_type='HLM', device="cpu"):
    # 원본 데이터 추출
    original_x = batch.x
    original_edge_index = batch.edge_index
    original_edge_attr = batch.edge_attr  # [num_edges, 3] 형태
    original_batch = batch.batch
    
    rng = np.random.default_rng(seed=seed)
    model.eval()
    phi_edges = []
    
    num_edges = original_edge_index.shape[1]
    
    if P is None:
        num_nodes = original_x.shape[0]
        max_num_edges = num_nodes * (num_nodes-1)
        P = num_edges / max_num_edges
    
    for j in range(num_edges):
        marginal_contrib = 0
        for i in range(M):
            # 랜덤 마스크 생성
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)
            
            # edge attribute 마스크 생성
            edge_attr_plus = create_plus_attr(pi, j, E_mask, E_z_mask, original_edge_attr)
            edge_attr_minus = create_minus_attr(pi, j, E_mask, E_z_mask, original_edge_attr)
            
            # edge j 포함 예측
            plus_batch = copy.deepcopy(batch)
            plus_batch.edge_attr = edge_attr_plus.to(device)
            plus_mlm, plus_res = model(plus_batch)
            plus_hlm = plus_mlm - plus_res
            
            # edge j 미포함 예측
            minus_batch = copy.deepcopy(batch)
            minus_batch.edge_attr = edge_attr_minus.to(device)
            minus_mlm, minus_res = model(minus_batch)
            minus_hlm = minus_mlm - minus_res
            
            if target_type == 'HLM':
                V_j_plus = plus_hlm
                V_j_minus = minus_hlm
            elif target_type == 'MLM':
                V_j_plus = plus_mlm
                V_j_minus = minus_mlm
            elif target_type == 'RES':
                V_j_plus = plus_res
                V_j_minus = minus_res
            # print(V_j_plus, V_j_minus)
            marginal_contrib += (V_j_plus - V_j_minus)
            
        phi_edges.append(marginal_contrib/M)
    
    return phi_edges

def create_plus_attr(pi, j, E_mask, E_z_mask, original_edge_attr):
    edge_attr = original_edge_attr.clone()
    zero_attr = torch.tensor([4, 0, 0])  # [3] 크기의 0 텐서
    selected_edge_index = torch.where(pi == j)[0].item()
    
    for k in range(len(E_z_mask)):
        if k <= selected_edge_index:
            if E_mask[pi[k]] == 1:
                edge_attr[pi[k]] = zero_attr
        else:
            if E_z_mask[pi[k]] == 1:
                edge_attr[pi[k]] = zero_attr
    return edge_attr

def create_minus_attr(pi, j, E_mask, E_z_mask, original_edge_attr):
    edge_attr = original_edge_attr.clone()
    zero_attr = torch.tensor([4, 0, 0])  # [3] 크기의 0 텐서
    selected_edge_index = torch.where(pi == j)[0].item()
    
    for k in range(len(E_z_mask)):
        if k < selected_edge_index:
            if E_mask[pi[k]] == 0:
                edge_attr[pi[k]] = zero_attr
        else:
            if E_z_mask[pi[k]] == 0:
                edge_attr[pi[k]] = zero_attr
    return edge_attr

# ============================================================
# 0) Matplotlib font setup (Korean/Unicode-safe)
# ============================================================

def _setup_matplotlib_font() -> None:
    """
    Configure matplotlib to use a Unicode-safe font (Korean-friendly if available).
    Also prevent unicode minus rendering issues, and improve PDF text compatibility.
    """
    preferred = [
        "NanumGothic", "Malgun Gothic", "AppleGothic",
        "Noto Sans CJK KR", "Noto Sans CJK JP",
        "Arial Unicode MS", "DejaVu Sans",
    ]
    avail = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in avail:
            matplotlib.rcParams["font.family"] = name
            break

    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


_setup_matplotlib_font()


# ============================================================
# 1) Wrapper for EdgeSHAPer (example implementation)
# ============================================================

class TaskWrapper(nn.Module):
    def __init__(self, base_model, task_idx, fp_vec=None, desc_vec=None, device=None, template_graph=None):
        super().__init__()
        self.base_model = base_model
        self.task_idx = int(task_idx)
        self.fp_vec = fp_vec
        self.desc_vec = desc_vec
        self.device = device if device is not None else next(base_model.parameters()).device
        self.template_graph = template_graph

    @torch.no_grad()
    def forward(self, x, edge_index, batch=None, edge_weight=None, **kwargs):
        src_batch = batch if (batch is not None) else getattr(self.template_graph, 'batch', None)
        batch_vec = self._ensure_batch(x, src_batch)

        graph = self._rebuild_graph(x, edge_index, edge_weight, batch_vec)

        _, outs = self.base_model({
            'fp': self.fp_vec,
            'graph': graph,
            'desc': self.desc_vec
        })

        z = outs[self.task_idx].view(-1).mean()
        logits2 = torch.stack([-z, z], dim=0).unsqueeze(0)  # (1,2)
        return logits2

    def _ensure_batch(self, x, src_batch=None):
        if src_batch is not None and int(src_batch.numel()) == int(x.size(0)):
            return src_batch.to(self.device)
        return torch.zeros(x.size(0), dtype=torch.long, device=self.device)

    def _rebuild_graph(self, x, edge_index, edge_weight, batch_vec):
        from torch_geometric.data import Data
        tg = self.template_graph

        edge_attr = getattr(tg, 'edge_attr', None) if tg is not None else None
        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            idxs = self._match_edge_order(getattr(tg, 'edge_index', None), edge_index) if tg is not None else None
            if idxs is not None: edge_attr = edge_attr[idxs]
            else: edge_attr = edge_attr[:edge_index.size(1)]

        g = Data(
            x=x.to(self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.to(self.device) if edge_attr is not None else None,
        )
        g.batch = batch_vec
        g.num_nodes = int(x.size(0))
        if edge_weight is not None:
            g.edge_weight = edge_weight.to(self.device)

        if tg is not None:
            for k in [
                'ring_mask','ring_index','nf_node','nf_ring','num_rings','n_nodes',
                'n_nfs','smiles','edge_to_bond','edge_uv'
            ]:
                if hasattr(tg, k):
                    setattr(g, k, getattr(tg, k))

            # ✅ FP 필드 복사 (dense 모드용)
            for k in ['morgan_fp', 'maccs_fp', 'rdit_fp']:
                if hasattr(tg, k):
                    v = getattr(tg, k)
                    v = v.to(self.device)
                    if v.dim() == 1:          
                        v = v.unsqueeze(0)
                    setattr(g, k, v)

            for k in ['fp', 'smil2vec']:
                if hasattr(tg, k):
                    setattr(g, k, getattr(tg, k).to(self.device))

        return g.to(self.device)

    def _match_edge_order(self, E0, E_new):
        if E0 is None: 
            return None
        try:
            import collections, torch
            buckets = collections.defaultdict(list)
            for i in range(E0.size(1)):
                u, v = int(E0[0, i]), int(E0[1, i])
                buckets[(u, v)].append(i)
            idxs = []
            for j in range(E_new.size(1)):
                u, v = int(E_new[0, j]), int(E_new[1, j])
                cand = buckets.get((u, v)) or buckets.get((v, u))
                if not cand:
                    return None
                idxs.append(cand[0])
            return torch.tensor(idxs, dtype=torch.long, device=E_new.device)
        except Exception:
            return None

# ============================================================
# 2) EdgeSHAPer for a single graph (merge directed edges)
# ============================================================

def explain_one_graph_edgeshaper(
    base_model: nn.Module,
    batch_item: Any,
    task_idx: int,
    target_class: int = 1,
    M: int = 128,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Explain a single molecular graph using EdgeSHAPer.

    Steps:
      1) Wrap the base model into an EdgeSHAPer-compatible callable
      2) Run Edgeshaper.explain() to obtain edge importances (directed edges)
      3) Map edges to RDKit bond indices (if available), otherwise infer via SMILES
      4) Merge (u,v) and (v,u) into undirected edges by summing scores

    Returns
    -------
    dict with:
      - edge_score: (E_uniq,) merged undirected edge scores
      - edge_uv: (E_uniq,2) undirected edge pairs (u < v)
      - edge_to_bonds: list[list[int]] bond indices corresponding to each undirected edge
    """
    device = device if device is not None else next(base_model.parameters()).device

    x = batch_item.x.to(device)
    edge_index = batch_item.edge_index.to(device)
    edge_weight = getattr(batch_item, "edge_weight", None)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    # Optional auxiliary vectors (depends on your MTMM forward)
    fp = getattr(batch_item, "smil2vec", None)
    desc = getattr(batch_item, "desc", None)

    wrapped = MTMMTaskWrapper(
        base_model=base_model,
        task_idx=task_idx,
        fp_vec=fp,
        desc_vec=desc,
        device=device,
        template_graph=batch_item,  # preserve batch + auxiliary fields
    ).to(device)

    es = Edgeshaper(wrapped, x, edge_index, edge_weight=edge_weight, device=str(device))
    edge_vals = np.asarray(
        es.explain(M=int(M), target_class=int(target_class), log_odds=False),
        dtype=float
    )

    # (A) Directed edge pairs from current graph
    uv_directed = edge_index.t().detach().cpu().numpy().astype(int)  # (E,2)

    # (B) Map edges to RDKit bond indices:
    #     If batch_item.edge_to_bond exists, use it; otherwise infer on the fly from SMILES.
    if hasattr(batch_item, "edge_to_bond"):
        e2b_directed = getattr(batch_item, "edge_to_bond").detach().cpu().numpy().astype(int)
        if e2b_directed.ndim > 1:
            e2b_directed = e2b_directed.reshape(-1)
    else:
        e2b_directed = []
        smi = getattr(batch_item, "smiles", None)
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and len(smi) > 0 else None

        for (u, v) in uv_directed:
            if mol is not None:
                b = mol.GetBondBetweenAtoms(int(u), int(v))
                e2b_directed.append(b.GetIdx() if b is not None else -1)
            else:
                e2b_directed.append(-1)

        e2b_directed = np.asarray(e2b_directed, dtype=int)

    # (C) Merge (u,v) and (v,u) into a single undirected key
    merged = defaultdict(lambda: {"score": 0.0, "bonds": set()})
    for (u, v), s, bidx in zip(uv_directed, edge_vals, e2b_directed):
        key = (int(min(u, v)), int(max(u, v)))  # undirected key
        merged[key]["score"] += float(s)
        if int(bidx) >= 0:
            merged[key]["bonds"].add(int(bidx))

    edge_uv_uniq: List[Tuple[int, int]] = []
    edge_score_uniq: List[float] = []
    bond_idx_list: List[List[int]] = []

    for (u, v), d in merged.items():
        edge_uv_uniq.append((u, v))
        edge_score_uniq.append(d["score"])
        bond_idx_list.append(sorted(list(d["bonds"])))

    return {
        "edge_score": np.asarray(edge_score_uniq, dtype=float),
        "edge_uv": np.asarray(edge_uv_uniq, dtype=int),
        "edge_to_bonds": bond_idx_list,
    }


# ============================================================
# 3) Subgraph / fragment extraction utilities
# ============================================================

def top_subgraphs_from_edges(
    mol: Chem.Mol,
    edge_scores: np.ndarray,
    edge_to_bonds: List[List[int]],
    top_frac: float = 0.2,
    sign: str = "pos",
) -> List[Tuple[List[int], List[int]]]:
    """
    Promote top-scoring edges (merged undirected edges) to connected subgraphs (fragments).

    Parameters
    ----------
    sign:
      - 'pos'  : keep edges with positive scores only
      - 'neg'  : keep edges with negative scores only
      - 'both' : ignore sign constraint

    Procedure
    ---------
      1) Determine threshold by absolute scores (quantile = 1-top_frac)
      2) Keep edges satisfying threshold (+ optional sign filter)
      3) Convert kept edges to RDKit bond indices
      4) Build atom adjacency over kept bonds
      5) Extract connected components (DFS) as subgraphs

    Returns
    -------
    List of (atoms, bonds) tuples, where:
      - atoms: list of atom indices in the component
      - bonds: list of RDKit bond indices in the component
    """
    if edge_scores is None or len(edge_scores) == 0:
        return []

    edge_scores = np.asarray(edge_scores, dtype=float)

    # Threshold based on absolute scores
    thr = float(np.quantile(np.abs(edge_scores), 1.0 - float(top_frac)))
    keep = (np.abs(edge_scores) >= thr)

    if sign == "pos":
        keep &= (edge_scores > 0)
    elif sign == "neg":
        keep &= (edge_scores < 0)
    elif sign == "both":
        pass
    else:
        raise ValueError("sign must be one of: 'pos', 'neg', 'both'")

    # Collect kept bonds
    kept_bonds: set[int] = set()
    for k, bonds in enumerate(edge_to_bonds):
        if keep[k]:
            kept_bonds.update([int(b) for b in bonds])

    if not kept_bonds:
        return []

    # Build atom adjacency from kept bonds
    adj = defaultdict(set)
    for bidx in kept_bonds:
        b = mol.GetBondWithIdx(int(bidx))
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[u].add(v)
        adj[v].add(u)

    # DFS connected components
    seen: set[int] = set()
    subgraphs: List[Tuple[List[int], List[int]]] = []

    for a in list(adj.keys()):
        if a in seen:
            continue
        stack = [a]
        comp_atoms: set[int] = set()

        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp_atoms.add(x)
            stack.extend(list(adj[x] - seen))

        # Bonds fully contained within comp_atoms
        comp_bonds: List[int] = []
        for bidx in kept_bonds:
            b = mol.GetBondWithIdx(int(bidx))
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if (u in comp_atoms) and (v in comp_atoms):
                comp_bonds.append(int(bidx))

        if comp_atoms:
            subgraphs.append((sorted(list(comp_atoms)), sorted(comp_bonds)))

    return subgraphs


def frag_smiles(mol: Chem.Mol, atoms: List[int], bonds: List[int]) -> str:
    """
    Convert a fragment defined by atom and bond indices into a fragment SMILES.
    """
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=list(atoms),
        bondsToUse=list(bonds),
        kekuleSmiles=True,
    )


def bond_scores_from_edges(edge_scores: np.ndarray, edge_to_bonds: List[List[int]]) -> Dict[int, float]:
    """
    Aggregate merged edge scores into per-bond scores by summing.
    """
    agg: Dict[int, float] = {}
    for s, bonds in zip(edge_scores, edge_to_bonds):
        for b in bonds:
            b = int(b)
            agg[b] = agg.get(b, 0.0) + float(s)
    return agg


# ============================================================
# 4) RDKit drawing helpers (bond highlighting)
# ============================================================

def score_to_rgb(score: float, vmax: float) -> Tuple[float, float, float]:
    """
    Map a signed score to an RGB color.
      - positive -> Blues
      - negative -> Reds
    Intensity is proportional to |score| / vmax.

    RDKit expects RGB values in [0,1].
    """
    if vmax <= 0:
        return (0.8, 0.8, 0.8)

    a = min(abs(float(score)) / float(vmax), 1.0)
    if score >= 0:
        r, g, b, _ = cm.Blues(0.5 + 0.5 * a)
    else:
        r, g, b, _ = cm.Reds(0.5 + 0.5 * a)

    return (float(r), float(g), float(b))


def mol_image_with_bond_scores(smiles: str, bond_scores: Dict[int, float], size: Tuple[int, int] = (420, 320)) -> Image.Image:
    """
    Render a molecule image with bond importance highlighted.

    Parameters
    ----------
    smiles: str
      Canonical SMILES
    bond_scores: dict[bond_idx -> score]
      Signed contributions (positive/negative)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Image.new("RGB", size, (255, 255, 255))

    if not bond_scores:
        return Draw.MolToImage(mol, size=size)

    vmax = max(abs(v) for v in bond_scores.values()) if bond_scores else 1.0
    highlight_bonds = list(bond_scores.keys())
    highlight_colors = {int(b): score_to_rgb(bond_scores[b], vmax) for b in bond_scores}

    img = Draw.MolToImage(
        mol,
        size=size,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_colors,
    )
    return img


# ============================================================
# 5) Figure creation: Panel A samples + Panel B fragment bars
# ============================================================

def pick_samples_for_panel_a(sample_results: List[Dict[str, Any]], k: int = 8) -> List[Dict[str, Any]]:
    """
    Pick top-k samples by total absolute edge contribution sum.
    """
    scored = []
    for d in sample_results:
        s = float(np.sum(np.abs(np.asarray(d["edge_scores"], dtype=float))))
        scored.append((s, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]


def draw_panel_a(
    sample_results: List[Dict[str, Any]],
    ncols: int,
    fig: plt.Figure,
    gs,
    title: str = "a",
    render_mol: bool = False,
    mol_img_size: Tuple[int, int] = (360, 260),
):
    """
    Draw a grid of sample molecules (Panel A).

    Parameters
    ----------
    gs:
      - Can be a GridSpec (already a grid) OR a SubplotSpec (a slot inside a bigger figure).
      - This function will create a sub-grid if needed.
    render_mol:
      - If True, render RDKit images with bond highlights (slower but informative).
      - If False, only show titles/placeholders (fast).
    """
    n = len(sample_results)
    if n == 0:
        ax = fig.add_subplot(gs if isinstance(gs, (GridSpec,)) else gs[0])
        ax.axis("off")
        ax.set_title(title)
        return

    ncols = max(1, int(ncols))
    nrows = max(1, int(math.ceil(n / ncols)))

    # If gs is a SubplotSpec, create a nested gridspec inside it
    if isinstance(gs, GridSpec):
        grid = gs
    else:
        grid = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs, wspace=0.25, hspace=0.35)

    for i, item in enumerate(sample_results[: nrows * ncols]):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(grid[r, c])

        smiles = item.get("smiles", "")
        ax.set_xticks([])
        ax.set_yticks([])

        if render_mol and smiles:
            edge_scores = np.asarray(item.get("edge_scores", []), dtype=float)
            edge_to_bonds = item.get("edge_to_bonds", [])
            bond_scores = bond_scores_from_edges(edge_scores, edge_to_bonds)
            img = mol_image_with_bond_scores(smiles, bond_scores, size=mol_img_size)
            ax.imshow(img)
            ax.axis("off")

        if smiles:
            ax.set_title(smiles, fontsize=9)

    fig.text(0.01, 0.99, title, fontsize=14, va="top", ha="left")


def aggregate_fragments(df_frag: pd.DataFrame, sign: str, agg: str = "weighted_mean", tasks: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aggregate fragment contributions across samples/tasks.

    Required columns in df_frag:
      - fragment (SMILES)
      - task (name)
      - sign ('pos'/'neg')
      - count (frequency)
      - mean_score (mean contribution per occurrence)

    agg:
      - 'weighted_mean': (sum(mean_score * count) / sum(count))
      - 'sum': sum(mean_score) per fragment
      - 'mean': mean(mean_score) per fragment
    """
    sub = df_frag[df_frag["sign"] == sign].copy()
    if tasks is not None:
        sub = sub[sub["task"].isin(tasks)]

    if sub.empty:
        return pd.DataFrame(columns=["fragment", "count_sum", "score_mean", "contrib"])

    g = sub.groupby("fragment", as_index=False).agg(
        count_sum=("count", "sum"),
        score_mean=("mean_score", "mean"),
    )

    if agg == "weighted_mean":
        tmp = sub.copy()
        tmp["_w"] = tmp["mean_score"] * tmp["count"]
        w = tmp.groupby("fragment", as_index=False).agg(w_sum=("_w", "sum"), c_sum=("count", "sum"))
        w["contrib"] = w["w_sum"] / w["c_sum"].clip(lower=1)
        out = g.merge(w[["fragment", "contrib"]], on="fragment", how="left")
    elif agg == "sum":
        s = sub.groupby("fragment", as_index=False)["mean_score"].sum().rename(columns={"mean_score": "contrib"})
        out = g.merge(s, on="fragment", how="left")
    else:
        out = g.rename(columns={"score_mean": "contrib"})

    # For 'neg', smaller values are "more destabilizing" (more negative),
    # so sorting ascending makes the most negative appear first.
    out = out.sort_values("contrib", ascending=(sign == "neg")).reset_index(drop=True)
    return out


def rdkit_smi_to_array(smi: str, w: int = 140, h: int = 110) -> np.ndarray:
    """
    Convert a SMILES into a small RGB array for embedding into matplotlib plots.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return np.ones((h, w, 3), dtype=np.uint8) * 255
    img = Draw.MolToImage(m, size=(w, h))
    return np.asarray(img)


def barplot_topk_with_frag_images(df_sorted: pd.DataFrame, sign: str, k: int = 10, ax=None, title: Optional[str] = None):
    """
    Draw top-k fragment bar plot and place fragment images under each bar.

    df_sorted should be output from aggregate_fragments():
      - fragment
      - contrib
    """
    top = df_sorted.head(int(k)).copy()
    if top.empty:
        if ax is None:
            plt.figure(figsize=(8, 2))
            plt.text(0.5, 0.5, "No fragments", ha="center", va="center")
            plt.axis("off")
            return
        ax.text(0.5, 0.5, "No fragments", ha="center", va="center")
        ax.axis("off")
        return

    frags = top["fragment"].tolist()
    vals = top["contrib"].values

    if ax is None:
        _, ax = plt.subplots(figsize=(min(18, 1.6 * len(frags)), 4))

    cmap = cm.Blues if sign == "pos" else cm.Reds
    denom = max(float(vals.max() - vals.min()), 1e-8)
    norm_vals = (vals - vals.min()) / denom
    colors = [cmap(0.4 + 0.55 * float(v)) for v in norm_vals]

    ax.bar(range(len(frags)), vals, color=colors)
    ax.set_xticks(range(len(frags)))
    ax.set_xticklabels(["" for _ in frags])
    ax.set_ylabel("Metabolic Stability Contribution", fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, pad=10)

    # Put molecule images under bars
    ymin, ymax = ax.get_ylim()
    y_img = ymin - (abs(ymin) * 0.25 if ymin < 0 else 0.25)
    ax.set_ylim(ymin, ymax)

    for i, smi in enumerate(frags):
        arr = rdkit_smi_to_array(smi, w=110, h=90)
        imagebox = OffsetImage(arr, zoom=0.9)
        ab = AnnotationBbox(
            imagebox, (i, y_img),
            xybox=(0, -30),
            xycoords="data",
            boxcoords="offset points",
            frameon=False,
        )
        ax.add_artist(ab)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    return ax


def create_full_figure(
    sample_results: List[Dict[str, Any]],
    df_frag: Optional[pd.DataFrame],
    out_path: str = "edgeshaper_figure.png",
    tasks: Optional[List[str]] = None,
    k_samples: int = 8,
    k_top: int = 10,
    agg: str = "weighted_mean",
    title: str = "Metabolic Stability – Fragment-Level Explanations (EdgeSHAPer)",
    show: bool = False,
    ncols: int = 4,
    dpi: int = 200,
    render_mol: bool = False,
):
    """
    Create a publication-style figure:
      - Panel A: sample-level explanations (grid)
      - Panel B: top-k destabilizing fragments
      - Panel C: top-k stabilizing fragments
    """
    sel = pick_samples_for_panel_a(sample_results or [], k=int(k_samples))
    n_a = len(sel)

    ncols = max(1, int(ncols))
    nrows_a = int(math.ceil(n_a / ncols)) if n_a > 0 else 0

    has_frag = (df_frag is not None) and (len(df_frag) > 0)
    rows_b = 2 if has_frag else 0
    total_rows = max(1, nrows_a + rows_b)

    fig_w = min(4.0 * max(1, ncols), 22)
    fig_h = max(3.6 * max(1, total_rows), 4.5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(total_rows, ncols)

    # Panel A
    if n_a > 0:
        draw_panel_a(sel, ncols=ncols, fig=fig, gs=gs[:nrows_a, :], title="a", render_mol=render_mol)
    else:
        ax_empty = fig.add_subplot(gs[0, :])
        ax_empty.text(0.5, 0.5, "No samples to display", ha="center", va="center", fontsize=13)
        ax_empty.axis("off")

    # Fragment panels
    if has_frag:
        df_neg = aggregate_fragments(df_frag, sign="neg", agg=agg, tasks=tasks)
        df_pos = aggregate_fragments(df_frag, sign="pos", agg=agg, tasks=tasks)

        ax_neg = fig.add_subplot(gs[nrows_a, :])
        barplot_topk_with_frag_images(df_neg, "neg", k=int(k_top), ax=ax_neg, title="Top Destabilizing Fragments")

        ax_pos = fig.add_subplot(gs[nrows_a + 1, :])
        barplot_topk_with_frag_images(df_pos, "pos", k=int(k_top), ax=ax_pos, title="Top Stabilizing Fragments")

    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    print(f"Saved: {out_path}")
    return out_path


# ============================================================
# 6) Export helpers: sample grid / fragment bars (per task)
# ============================================================

def save_samples_grid(sample_results: List[Dict[str, Any]], out_path: str, ncols: int = 4, dpi: int = 200, render_mol: bool = False):
    """
    Save only Panel A (sample grid) as a single image.
    """
    sel = pick_samples_for_panel_a(sample_results or [], k=len(sample_results))
    n = len(sel)
    ncols = max(1, int(ncols))
    nrows = max(1, int(math.ceil(n / ncols)))

    fig = plt.figure(figsize=(min(4.0 * ncols, 22), 3.6 * nrows))
    gs = fig.add_gridspec(nrows, ncols)

    draw_panel_a(sel, ncols=ncols, fig=fig, gs=gs, title="a", render_mol=render_mol)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved samples grid: {out_path}")


def _wrap(s: str, width: int = 28) -> str:
    """
    Wrap long strings for nicer axis labels.
    """
    s = str(s)
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s


def save_fragment_bars(df_frag: pd.DataFrame, out_dir: str, tasks: List[str], k_top: int = 10, dpi: int = 160):
    """
    Save fragment bar plots per task.
    Supports two modes:
      - If df_frag has 'sign' column: save separate plots for pos/neg
      - Otherwise: save single top-k plot per task
    """
    os.makedirs(out_dir, exist_ok=True)

    if "task" not in df_frag.columns:
        raise ValueError("df_frag must include a 'task' column.")

    has_sign = ("sign" in df_frag.columns)

    def _topk(sub: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        # Guess score column
        score_col = None
        for c in ["mean_abs_shap", "mean_shap", "score", "shap", "contrib", "mean_score"]:
            if c in sub.columns:
                score_col = c
                break
        if score_col is None:
            raise ValueError("df_frag must include a score column (e.g., mean_score / contrib / shap).")
        return sub.sort_values(score_col, ascending=False).head(int(k_top)), score_col

    for t in tasks:
        sub = df_frag[df_frag["task"] == t].copy()
        if sub.empty:
            continue

        if has_sign:
            for sign in ["pos", "neg"]:
                ss = sub[sub["sign"] == sign].copy()
                if ss.empty:
                    continue
                ss, score_col = _topk(ss)

                label_col = "fragment" if "fragment" in ss.columns else ss.columns[0]
                ss[label_col] = ss[label_col].map(lambda x: _wrap(x, 28))

                fig_h = min(6.0, max(3.0, 0.45 * len(ss) + 1.5))
                fig, ax = plt.subplots(figsize=(8.5, fig_h))
                ax.barh(ss[label_col].values[::-1], ss[score_col].values[::-1])

                ax.set_title(f"{t} | top-{len(ss)} fragments ({sign})")
                ax.set_xlabel(score_col)
                ax.grid(True, axis="x", alpha=0.3)

                fig.set_constrained_layout(True)

                out_png = os.path.join(out_dir, f"edgeshaper_frag_{t}_{sign}.png")
                fig.savefig(out_png, dpi=int(dpi), pad_inches=0.2)
                plt.close(fig)

        else:
            sub, score_col = _topk(sub)

            label_col = "fragment" if "fragment" in sub.columns else sub.columns[0]
            sub[label_col] = sub[label_col].map(lambda x: _wrap(x, 28))

            fig_h = min(6.0, max(3.0, 0.45 * len(sub) + 1.5))
            fig, ax = plt.subplots(figsize=(8.5, fig_h))
            ax.barh(sub[label_col].values[::-1], sub[score_col].values[::-1])

            ax.set_title(f"{t} | top-{len(sub)} fragments")
            ax.set_xlabel(score_col)
            ax.grid(True, axis="x", alpha=0.3)

            fig.set_constrained_layout(True)

            out_png = os.path.join(out_dir, f"edgeshaper_frag_{t}.png")
            fig.savefig(out_png, dpi=int(dpi), pad_inches=0.2)
            plt.close(fig)

    print(f"[OK] Saved fragment bar plots into: {out_dir}")


# ============================================================
# 7) Sample-wise explanation collection (Panel A input)
# ============================================================
def _collect_sample_results_for_panel(
    model: nn.Module,
    test_loader: GeometricDataLoader,
    tasks: List[str],
    device: torch.device,
    task_for_panel: str = "human",
    max_samples: Optional[int] = None,
    M_explain: int = 128,
) -> List[Dict[str, Any]]:
    """
    Collect per-sample EdgeSHAPer results for a selected task (Panel A).
    """
    assert task_for_panel in tasks, f"task_for_panel must be one of: {tasks}"
    t_idx = tasks.index(task_for_panel)

    out_list: List[Dict[str, Any]] = []
    model.eval()

    # NOTE: Edgeshaper itself will evaluate many forward passes; do NOT wrap everything with no_grad()
    for batch in test_loader:
        batch = batch.to(device)
        for item in batch.to_data_list():
            # Predicted probability for the selected task
            with torch.no_grad():
                _, outs = model({
                    "fp": getattr(item, "smil2vec", None),
                    "graph": item,
                    "desc": getattr(item, "desc", None),
                })
                y_pred = torch.sigmoid(outs[t_idx]).view(-1).mean().item()

                y_true = None
                if getattr(item, "y", None) is not None:
                    yv = item.y.view(-1)
                    if yv.numel() > t_idx and float(yv[t_idx]) != -1.0:
                        y_true = float(yv[t_idx])

            # EdgeSHAPer explanation (this will run multiple model calls internally)
            xai = explain_one_graph_edgeshaper(
                base_model=model,
                batch_item=item,
                task_idx=t_idx,
                target_class=1,
                M=int(M_explain),
                device=device,
            )

            out_list.append({
                "smiles": getattr(item, "smiles", ""),
                "y_true": y_true,
                "y_pred": y_pred,
                "edge_scores": xai["edge_score"],
                "edge_to_bonds": xai["edge_to_bonds"],
            })

            if (max_samples is not None) and (len(out_list) >= int(max_samples)):
                return out_list

    return out_list


# ============================================================
# 8) Cross-species common fragment analysis
# ============================================================
def analyze_species_common(
    model: nn.Module,
    loader: GeometricDataLoader,
    task_names: Tuple[str, ...] = ("human", "rat", "mouse"),
    M: int = 128,
    top_frac: float = 0.2,
    device: Optional[torch.device] = None,
    posneg: Tuple[str, ...] = ("pos", "neg"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze fragments across multiple tasks (e.g., human/rat/mouse) and compute:
      - df: per-task fragment frequency and mean score
      - df_common: fragments that appear in ALL tasks (intersection), split by sign

    Notes
    -----
    - This implementation defines fragment_score as the mean edge_score across edges whose bond set
      intersects the fragment bond set (simple heuristic).
    - If you want a stricter definition (e.g., sum of bond scores within fragment),
      modify frag_score computation below.
    """
    device = device if device is not None else next(model.parameters()).device

    # stats[task][sign]['freq'][frag] = count
    # stats[task][sign]['score_sum'][frag] = sum(fragment_score)
    stats = {t: {sgn: {"freq": Counter(), "score_sum": Counter()} for sgn in posneg} for t in task_names}

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        for item in batch.to_data_list():
            smi = getattr(item, "smiles", "")
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # Explain each task for this molecule
            for ti, t in enumerate(task_names):
                out = explain_one_graph_edgeshaper(
                    base_model=model,
                    batch_item=item,
                    task_idx=ti,
                    target_class=1,
                    M=int(M),
                    device=device,
                )
                edge_scores = out["edge_score"]
                edge_to_bonds = out["edge_to_bonds"]

                # Extract subgraphs for each sign
                for sgn in posneg:
                    subgs = top_subgraphs_from_edges(
                        mol,
                        edge_scores=edge_scores,
                        edge_to_bonds=edge_to_bonds,
                        top_frac=float(top_frac),
                        sign=sgn,
                    )

                    for atoms, bonds in subgs:
                        frag = frag_smiles(mol, atoms, bonds)

                        # Fragment score heuristic: mean edge_score over edges touching any bond in this fragment
                        touched_scores = []
                        bond_set = set(int(b) for b in bonds)
                        for k, bs in enumerate(edge_to_bonds):
                            if any(int(b) in bond_set for b in bs):
                                touched_scores.append(float(edge_scores[k]))

                        frag_score = float(np.mean(touched_scores)) if touched_scores else 0.0

                        stats[t][sgn]["freq"][frag] += 1
                        stats[t][sgn]["score_sum"][frag] += frag_score

    # Build dataframe
    frames = []
    for t in task_names:
        for sgn in posneg:
            freq = stats[t][sgn]["freq"]
            ssum = stats[t][sgn]["score_sum"]
            rows = [(frag, t, sgn, freq[frag], ssum[frag] / max(freq[frag], 1)) for frag in freq]
            frames.append(pd.DataFrame(rows, columns=["fragment", "task", "sign", "count", "mean_score"]))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["fragment", "task", "sign", "count", "mean_score"]
    )

    # Intersection fragments across all tasks (per sign)
    commons = []
    for sgn in posneg:
        sub = df[df["sign"] == sgn]
        if sub.empty:
            continue

        sets = []
        for t in task_names:
            sets.append(set(sub[sub["task"] == t]["fragment"]))
        common_frags = set.intersection(*sets) if len(sets) > 1 else sets[0]

        commons.append(sub[sub["fragment"].isin(common_frags)].copy())

    df_common = pd.concat(commons, ignore_index=True) if commons else pd.DataFrame(columns=df.columns)
    return df, df_common
