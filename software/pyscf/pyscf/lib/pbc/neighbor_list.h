#ifndef HAVE_DEFINED_NEIGHBOR_LIST_H
#define HAVE_DEFINED_NEIGHBOR_LIST_H
typedef struct NeighborPair_struct {
    int nimgs;
    int *Ls_list;
    double *q_cond;
    double *center;
} NeighborPair;

typedef struct NeighborList_struct {
    int nish;
    int njsh;
    int nimgs;
    NeighborPair **pairs;
} NeighborList;

typedef struct NeighborListOpt_struct {
    NeighborList *nl;
    int (*fprescreen)(int *shls, struct NeighborListOpt_struct *opt);
} NeighborListOpt;

int NLOpt_noscreen(int* shls, NeighborListOpt* opt);
#endif
