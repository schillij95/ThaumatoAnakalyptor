// node_structs.h
#ifndef NODE_STRUCTS_H
#define NODE_STRUCTS_H

struct Edge {
    unsigned int target_node;
    float certainty;
    float certainty_factor = 1.0f;
    float certainty_factored;
    float k;
    bool same_block;
    bool fixed = false;
};

struct Node {
    float z;                     // Z-coordinate of the node
    float f_init;                // Initial value of f for this node
    float f_tilde;               // Current value of f_tilde for this node (used in BP updates)
    float f_star;                // The computed final value of f for this node
    bool gt;                     // Indicates if this node has ground truth available
    float gt_f_star;             // Ground truth f_star value, if available
    bool deleted;                // If the node is marked as deleted
    bool fixed;                  // Whether this node is fixed or not
    float fold;                  // Folding state for the node

    Edge* edges;                 // Pointer to an array of edges (dynamic array)
    int num_edges;               // Number of edges connected to this node
};

#endif
