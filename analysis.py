import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
import powerlaw
from random import randint
import numpy as np

#TODO Use small increments of percents going up to 50% or less for country removals
#TODO Ignore "Digital" and "Unknwown" in node removal for "Ships From"

def create_bpt_network(rows,s_ctg,t_ctg,num_src):
    #TODO Figure out if links to Unknown should be shown
    g = ig.Graph.TupleList(list(set(rows.itertuples(index=False,name=None)))) if isinstance(rows,pd.DataFrame) else ig.Graph.TupleList(rows)
    print(f"Number of {s_ctg}: {num_src}")
    print(f"Number of {t_ctg}: {g.vcount()-num_src}")
    print(f"Number of edges: {g.ecount()}")
    return g

def create_network(df):
    g = ig.Graph.TupleList(list(set(df.itertuples(index=False,name=None))))
    print(f"|N|={g.vcount()}") 
    print(f"|L|={g.ecount()}")
    return g

def plot_degr_distr(degrees, network_name):
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), density=True, alpha=0.75, color='b')

    plt.xlabel("Degree")
    plt.ylabel("Frequency (Normalized)")
    plt.title(f"{network_name} Degree Distribution")

    plt.yscale("log") 
    plt.xscale("log")
    plt.savefig(f"degree_distributions/{network_name.lower().replace(' ','_')}_degr_distr.png")
    plt.show()
    plt.close()

def cmpr_distr(degrees,network_name):
    fit = powerlaw.Fit(degrees)
    print(f"Power-law exponent: {fit.alpha}")
    print(f"Minimum degree for power-law behavior: {fit.xmin}")
    R, p = fit.distribution_compare('power_law', 'exponential')
    print(f"Power-law vs exponential: R={R}, p-value: {p}")
    R_dpln, p_dpln = fit.distribution_compare('power_law', 'lognormal_positive')
    print(f"Power-law vs Double Pareto Log-Normal: R={R_dpln}, p={p_dpln}")
    fig = fit.plot_ccdf(linewidth=2, label='Empirical Data')
    fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power-Law Fit')
    #fit.exponential.plot_ccdf(ax=fig, color='g', linestyle=':', label='Exponential Fit')
    #fit.lognormal.plot_ccdf(ax=fig, color='b', linestyle='-.', label='Log-Normal Fit')
    #fit.truncated_power_law.plot_ccdf(ax=fig, color='b', linestyle='-.', label='Truncated Power-Law Fit')
    plt.legend()
    plt.savefig(f"degree_distributions/{network_name.lower().replace(' ','_')}_power_law_fit")
    plt.show()
    plt.close()

def analyze_degr_distr(g:ig.Graph, srcs=None, network_name=""):
    degrees = g.degree(srcs) if srcs else g.degree()
    print(f"Average degree: {sum(degrees)*1.0/len(degrees)}")
    top = sorted(zip(srcs if srcs else g.vs["name"],degrees),key=lambda x:x[1],reverse=True)[:10]
    print(f"Highest degree node:\n {top[0]}")
    plot_degr_distr(degrees, network_name)
    cmpr_distr(degrees,network_name)

def analyze_btwness(g:ig.Graph,srcs=None,cutoff=None):
    btwness = g.betweenness(srcs,cutoff=cutoff) if srcs else g.betweenness()
    print(f"Average betweenness: {sum(btwness)*1.0/len(btwness)}")
    top = sorted(zip(srcs if srcs else g.vs,btwness),key=lambda x:x[1],reverse=True)[:10]
    print(f"Highest betweenness nodes:\n {top}")

def analyze_eigen_centrality(g:ig.Graph,srcs=None):
    eigen_centrality = g.eigenvector_centrality(srcs) if srcs else g.eigenvector_centrality()
    print(f"Average eigenvector centrality: {sum(eigen_centrality)*1.0/len(eigen_centrality)}")
    top = sorted(zip(srcs if srcs else g.vs,eigen_centrality),key=lambda x:x[1],reverse=True)[:10]
    print(f"Highest eigenvector centrality nodes:\n {top}")

def get_lrgst_comp(g:ig.Graph):
    clust_obj = ig.Graph.components(g,"weak")
    #print(clust_obj.summary(verbosity=1)) #prints membership of clusters
    sizes = clust_obj.sizes()
    """for i, clust_size in enumerate(sizes):
        print(clust_size)
        if i!=0: #not printing the largest cluster's members assuming list is sorted by non-increasing cluster order
            print(g.vs(clust_obj.__getitem__(i))["name"]) #prints membership of clusters in descending order of cluster size"""
    return max(sizes), clust_obj.giant()

def simulate_random_attack(g:ig.Graph,targets:list=None,num_simulations=10,checkpoints=[0.10,0.25,0.5,0.75,0.9],network_name=""):
    """
    @param g : network
    @param targets : list of unique target nodes (list for random index removal)
    @param num_simulations
    @param checkpoints : proportion of nodes needed to be removed before recording stats
    """
    lcc = [[0 for cp in range(len(checkpoints)+1)] for sim in range(num_simulations)]
    btwness = [[0 for cp in range(len(checkpoints)+1)] for sim in range(num_simulations)]
    #egn_centr
    for i in range(num_simulations):
        lcc[i][0] = get_lrgst_comp(g)[0]
        btw = g.betweenness()
        btwness[i][0] = sum(btw)/len(btw)
    for sim in range(num_simulations):
        nodes = None
        n = None
        tgts = None
        if targets:
            nodes = set(g.vs["name"])
            n = len(tgts)
            tgts = [tgt for tgt in targets]
        else:
            nodes = list(g.vs["name"])
            n = len(nodes)
        checkpoint = 0
        while checkpoint<len(checkpoints):
            if targets:
                while len(tgts)>n*(1-checkpoints[checkpoint]):
                    rm_ind = randint(0,len(tgts)-1)
                    node_name = tgts.pop(rm_ind)
                    nodes.remove(node_name)
            else:
                while len(nodes)>n*(1-checkpoints[checkpoint]):
                    rm_ind = randint(0,len(nodes)-1)
                    nodes.pop(rm_ind)
            lrgst_comp_size, lrgst_cmp = get_lrgst_comp(g.subgraph(nodes))
            lcc[sim][checkpoint+1] = lrgst_comp_size
            btw = lrgst_cmp.betweenness()
            btwness[sim][checkpoint+1] = sum(btw)/len(btw)
            checkpoint+=1
    mean_lcc = np.mean(lcc, axis=0)
    std_dev_lcc = np.std(lcc, axis=0)
    checkpoints = [0] + checkpoints
    plt.figure(figsize=(8, 5))
    plt.errorbar(checkpoints, mean_lcc, yerr=std_dev_lcc, fmt='-o', capsize=5, label=f"Mean LCC Size Across {num_simulations} Simulations")
    plt.xlabel("Percentage of Nodes Removed")
    plt.xticks(checkpoints)
    plt.ylabel("Size of Largest Connected Component")
    plt.title("Impact of Node Removal on Largest Connected Component")
    plt.legend()
    plt.grid()
    plt.savefig(f"simulation_results/{network_name.lower().replace(' ','_')}_random_removal_lcc.png")
    plt.show()
    plt.close()

    mean_btwness = np.mean(btwness, axis=0)
    std_dev_btwness = np.std(btwness, axis=0)
    plt.figure(figsize=(8, 5))
    plt.errorbar(checkpoints, mean_btwness, yerr=std_dev_btwness, fmt='-o', capsize=5, label=f"Mean Betweenness Size Across {num_simulations} Simulations")
    plt.xlabel("Percentage of Nodes Removed")
    plt.xticks(checkpoints)
    plt.ylabel("Betweenness")
    plt.title("Impact of Node Removal on Betweenness")
    plt.legend()
    plt.grid()
    plt.savefig(f"simulation_results/{network_name.lower().replace(' ','_')}_random_removal_betweenness.png")
    plt.show()
    plt.close()

if __name__=="__main__":
    #Agora Vendor-Prod Category
    """agora_df = pd.read_csv("Agora.csv", usecols=["Vendor","Category"], dtype=str)
    vendors = set(agora_df["Vendor"])
    agora_network = create_bpt_network(agora_df,"Vendor","Category",len(vendors))
    analyze_degr_distr(agora_network,network_name="Agora v-pc")
    #simulate_random_attack(agora_network,vendors,network_name="Agora v-pc")
    #print(get_lrgst_comp(agora_network))
    
    ig.plot(agora_network, target="networks/agora_vendor_category_network.png")
    vend_lst = list(vendors)
    #analyze_degr_distr(agora_network,vend_lst,"Agora v-pc Vendor")"""
    """analyze_btwness(agora_network,vend_lst)
    analyze_eigen_centrality(agora_network,vend_lst)"""
    
    #market_data_obfuscated.csv cleaning
    
    """mkt_data_obf_df = pd.read_csv("market_data_obfuscated.csv", usecols=["Market","Product Category","Ships From","Vendor Username"],dtype=str)
    mkt_data_obf_df = mkt_data_obf_df[mkt_data_obf_df["Product Category"]!="intoxicants"] #intoxicants category is inaccurate so it is ignored
    """
    #Country extension functions
    """countries = set() 
    def extend_countries_lst(vndr:str,vndr_cntrs:set,rgn_expansion_lst:list,exclude:list):
        for country in rgn_expansion_lst:
            if country not in exclude:
                vndr_cntrs.add((vndr,country))
                countries.add(country)

    def expand_regions(df):
        vndr_cntrs = set()
        #constants below are most likely shipping locations for broad regions from 2014-2015
        LAT_AMER = ["Mexico", "Brazil", "Colombia", "Argentina", "Chile", 
        "Peru", "Venezuela", "Ecuador", "Guatemala", "Bolivia"]
        EU = ["UK", "Germany", "Netherlands", "France", "Italy", "Spain", 
            "Belgium", "Austria", "Poland", "Sweden", "Czech_Republic", 
            "Ireland", "Denmark", "Finland", "Portugal", "Greece", "Romania"]
        EUROPE = ["UK", "Germany", "Netherlands", "France", "Italy", 
        "Spain", "Belgium", "Austria", "Poland", "Sweden", "Czech_Republic", 
        "Ireland", "Denmark", "Finland", "Portugal", "Greece", 
        "Romania", "Hungary", "Bulgaria", "Slovakia", "Croatia", 
        "Slovenia", "Estonia", "Latvia", "Lithuania"]
        ANTILLES = ["Cuba", "Dominican_Republic", "Haiti", "Jamaica", 
        "Puerto_Rico", "Barbados", "Trinidad_and_Tobago", 
        "Saint_Lucia", "Saint_Vincent_and_the_Grenadines", 
        "Antigua_and_Barbuda", "Saint_Kitts_and_Nevis"]
        ASIA = ["China", "India", "Pakistan", "Afghanistan", "Thailand",
        "Vietnam", "Cambodia", "Indonesia", "Malaysia", "Singapore", 
        "Philippines", "Japan", "South_Korea", "Taiwan", "Sri_Lanka"]
        for _, row in df.iterrows():
            for rgn in row["Ships From"].split(' '):
                exclude = rgn.split('-') #minus used once in the dataset Latin America to exclude a country so for more complex example, logic may have to be edited
                if exclude:
                    exclude = exclude[1:]
                if "Latin_America" in rgn:
                    extend_countries_lst(row["Vendor Username"],vndr_cntrs,LAT_AMER,exclude)
                if "EU" in rgn:
                    extend_countries_lst(row["Vendor Username"],vndr_cntrs,EU,exclude)
                if "Europe" in rgn:
                    extend_countries_lst(row["Vendor Username"],vndr_cntrs,EUROPE,exclude)
                if "Antilles" in rgn:
                    extend_countries_lst(row["Vendor Username"],vndr_cntrs,ANTILLES,exclude)
                if "Asia" in rgn:
                    extend_countries_lst(row["Vendor Username"],vndr_cntrs,ASIA,exclude)
                else:
                    vndr_cntrs.add((row["Vendor Username"],rgn))
                    if rgn not in ("Digital","Unknown"):
                        countries.add(rgn)
        return list(vndr_cntrs)"""

    #Agora Ship From-Vendor
    """agora_df = mkt_data_obf_df[mkt_data_obf_df["Market"]=="Agora"]
    agora_network = create_bpt_network(expand_regions(agora_df),"Ships From","Vendor Username",len(countries))
    """
    #ig.plot(agora_network, target="networks/agora_country_vendor_network.png")
    """name_to_index = {v['name']: v.index for v in agora_network.vs}
    print(f"Degree of Unknown: {agora_network.degree(name_to_index.get("Unknown",0))}")
    print(f"Degree of Digital: {agora_network.degree(name_to_index.get("Digital",0))}")"""
    #analyze_degr_distr(agora_network, network_name="Agora sf-v")
    #simulate_random_attack(agora_network,countries,num_simulations=50,network_name="Agora sf-v")
    """cntr_lst = list(countries)
    analyze_degr_distr(agora_network,cntr_lst, "Agora sf-v Country")
    analyze_btwness(agora_network,cntr_lst)"""
    
    #filter for Silk Road 2
    """silk_rd2_df = mkt_data_obf_df[mkt_data_obf_df["Market"]=="Silk Road 2"]"""

    #Silk Road 2 Vendor-Prod Category
    """vendors = set(silk_rd2_df["Vendor Username"])
    silk_rd2_network = create_bpt_network(silk_rd2_df[["Vendor Username", "Product Category"]],"Vendor Username", "Product Category",len(vendors))
    ig.plot(silk_rd2_network, target="networks/silk_road_2_vendor_category_network.png")
    analyze_degr_distr(silk_rd2_network,network_name="Silk Road 2 v-pc")
    simulate_random_attack(silk_rd2_network,vendors,network_name="Silk Road 2 v-pc")
    vend_lst = list(vendors)
    analyze_degr_distr(silk_rd2_network,vend_lst,"Silk Road 2 v-pc Vendor")"""
    #analyze_btwness(silk_rd2_network,vend_lst)
    
    #Silk Road 2 Ship From-Vendor Network
    """countries = set()
    silk_rd2_network = create_bpt_network(expand_regions(silk_rd2_df),"Ships From","Vendor Username",len(countries))
    name_to_index = {v['name']: v.index for v in silk_rd2_network.vs}
    print(f"Degree of Unknown: {silk_rd2_network.degree(name_to_index.get("Unknown",0))}")
    print(f"Degree of Digital: {silk_rd2_network.degree(name_to_index.get("Digital",0))}")"""
    #ig.plot(silk_rd2_network, target="networks/silk_road_2_country_vendor_network.png")
    #analyze_degr_distr(silk_rd2_network,network_name="Silk Road 2 sf-v")
    #simulate_random_attack(silk_rd2_network,countries,num_simulations=50,network_name="Silk Road 2 sf-v")
    """cntr_lst = list(countries)
    analyze_degr_distr(silk_rd2_network,cntr_lst,"Silk Road 2 sf-v Country")
    analyze_btwness(silk_rd2_network,cntr_lst)"""

    #Evolution User-User network
    """evol_df = pd.read_csv("evolution_data/network/edges-2014-1.tsv",usecols=["Source","Target"],sep="\t",dtype=str)
    for i in range(2,13):
        evol_df = pd.concat([evol_df,pd.read_csv(f"evolution_data/network/edges-2014-{i}.tsv",usecols=["Source","Target"],sep="\t",dtype=str)],ignore_index=True)
    for i in range(1,4):
        evol_df = pd.concat([evol_df,pd.read_csv(f"evolution_data/network/edges-2015-{i}.tsv",usecols=["Source","Target"],sep="\t",dtype=str)],ignore_index=True)
    evol_network = create_network(evol_df)
    #analyze_degr_distr(evol_network,srcs=None,network_name="Evolution User Network")
    #ig.plot(evol_network, target="networks/evolution_user_network.png")
    simulate_random_attack(evol_network,num_simulations=3,network_name="Evolution User Network")"""
    #analyze_btwness(evol_network,cutoff=2)

