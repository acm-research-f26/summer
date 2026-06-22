#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <chrono>
#include <cassert>
#include <string>
#include <utility>
#include <cstdio>

// ---- Memory Layout Structures ----
struct EdgeNode {
    int neighbor;
    EdgeNode* next;
};

struct PointerGraph {
    int num_vertices = 0;
    std::vector<EdgeNode*> vertices;
    std::vector<int> degrees;
};

struct CSRGraph {
    int num_vertices = 0;
    std::vector<int> offsets;
    std::vector<int> edges;
    std::vector<int> degrees;
};

// ---- Graph Parsing Engine (Bypassing MinGW Bug) ----
void build_graphs(const std::string& filename, PointerGraph& p_graph, CSRGraph& csr_graph) {
    FILE* infile = std::fopen(filename.c_str(), "r");
    if (!infile) {
        std::cerr << "Error: Could not open " << filename << ". Run the Python script first!\n";
        return;
    }

    int u, v;
    int max_vertex = 0;
    std::vector<std::pair<int, int>> edge_list;

    while (std::fscanf(infile, "%d %d", &u, &v) == 2) {
        edge_list.push_back({u, v});
        if (u > max_vertex) max_vertex = u;
        if (v > max_vertex) max_vertex = v;
    }
    std::fclose(infile);
    
    int V = max_vertex + 1;
    std::cout << "Graph structure parsed. V = " << V << " | Total Edges = " << edge_list.size() << "\n";

    p_graph.num_vertices = V;
    p_graph.vertices.resize(V, nullptr);
    p_graph.degrees.resize(V, 0);

    csr_graph.num_vertices = V;
    csr_graph.degrees.resize(V, 0);

    // 1. Build Pointer Graph
    for (const auto& edge : edge_list) {
        u = edge.first;
        v = edge.second;
        
        EdgeNode* new_node = new EdgeNode{v, p_graph.vertices[u]};
        p_graph.vertices[u] = new_node;
        p_graph.degrees[u]++;
        csr_graph.degrees[u]++; 
    }

    // 2. Build CSR Graph 
    csr_graph.offsets.resize(V + 1, 0);
    for (int i = 0; i < V; ++i) {
        csr_graph.offsets[i + 1] = csr_graph.offsets[i] + csr_graph.degrees[i];
    }
    csr_graph.edges.resize(csr_graph.offsets[V], 0);

    std::vector<int> current_offset_pos = csr_graph.offsets;
    for (const auto& edge : edge_list) {
        u = edge.first;
        v = edge.second;
        int dest_idx = current_offset_pos[u]++;
        csr_graph.edges[dest_idx] = v;
    }
}

// ---- Pointer BFS with Live Logging ----
std::vector<bool> run_pointer_bfs(const PointerGraph& graph, int start_node, std::ofstream& trace_file) {
    std::vector<bool> visited(graph.num_vertices, false);
    std::queue<int> q;

    if (graph.num_vertices == 0) return visited;

    visited[start_node] = true;
    q.push(start_node);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        int u_degree = graph.degrees[u];
        EdgeNode* curr = graph.vertices[u];
        while (curr != nullptr) {
            trace_file << "POINTER," << u_degree << "," << reinterpret_cast<uintptr_t>(curr) << "\n";
            int v = curr->neighbor;
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
            }
            curr = curr->next;
        }
    }
    return visited;
}

// ---- CSR BFS with Live Logging ----
std::vector<bool> run_csr_bfs(const CSRGraph& graph, int start_node, std::ofstream& trace_file) {
    std::vector<bool> visited(graph.num_vertices, false);
    std::queue<int> q;

    if (graph.num_vertices == 0) return visited;

    visited[start_node] = true;
    q.push(start_node);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        int u_degree = graph.degrees[u];
        int start_idx = graph.offsets[u];
        int end_idx = graph.offsets[u + 1];

        for (int i = start_idx; i < end_idx; ++i) {
            trace_file << "CSR," << u_degree << "," << reinterpret_cast<uintptr_t>(&graph.edges[i]) << "\n";
            int v = graph.edges[i];
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
    return visited;
}

// ---- Main Pipeline Execution ----
int main() {
    PointerGraph p_graph;
    CSRGraph csr_graph;

    build_graphs("edges.txt", p_graph, csr_graph);
    if (p_graph.num_vertices == 0) return 1;

    std::ofstream trace_file("trace.csv");
    if (!trace_file.is_open()) {
        std::cerr << "Error: Could not create trace.csv\n";
        return 1;
    }
    trace_file << "layout,degree,address\n";

    std::cout << "Running Pointer BFS...\n";
    std::vector<bool> pointer_results = run_pointer_bfs(p_graph, 0, trace_file);

    std::cout << "Running CSR BFS...\n";
    std::vector<bool> csr_results = run_csr_bfs(csr_graph, 0, trace_file);

    trace_file.close();
    std::cout << "Trace file 'trace.csv' written.\n";

    std::cout << "Verifying topological equivalence... ";
    assert(pointer_results == csr_results);
    std::cout << "PASSED!\n";

    // Clean up heap pointer memory
    for (int i = 0; i < p_graph.num_vertices; ++i) {
        EdgeNode* curr = p_graph.vertices[i];
        while (curr != nullptr) {
            EdgeNode* next = curr->next;
            delete curr;
            curr = next;
        }
    }
    return 0;
}