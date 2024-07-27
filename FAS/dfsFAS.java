/*
    FAS via DFS back edges

    Copyright (C) 2016 by
    Michael Simpson <simpsonm@uvic.ca>
    All rights reserved.
    BSD license.
*/

import java.io.IOException;
import java.util.BitSet;
import java.util.Deque;
import java.util.ArrayDeque;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

public class dfsFAS {

	String basename; // input graph basename
   	ImmutableGraph G; // graph G
	int n; // number of vertices in G
	int fas; // number of edges in fas
	int[] color; // color nodes to determine back edges: 0=white, 1=grey, 2=black
		
	// load graph and initialize class variables
	public dfsFAS(String basename) throws Exception {
		this.basename = basename;
		
		System.out.println("Loading graph...");
		G = ImmutableGraph.load(basename); //We need random access
		System.out.println("Graph loaded");
		
		n = G.numNodes();
		System.out.println("n="+n);
		System.out.println("e="+G.numArcs());	
		
		color = new int[n];
		fas = 0;
	}
	
	public void DFS(int v) throws Exception {
     	color[v] = 1; // v discovered
     	
     	int[] v_neighbors = G.successorArray(v);
     	int v_deg = G.outdegree(v);
     	int w;
			
     	for(int i = 0; i < v_deg; i++) { // explore edge
      		w = v_neighbors[i];
				
      		if(v == w) { // Self-loop, ignore
     			continue;
      		}
      		
      		if (color[w] == 0) {
          		DFS(w);
      		} else if (color[w] == 1) {
          		fas++;
      		}
      	}
      	
      	color[v] = 2; // v finished
	}
	
	public void computeFAS() throws Exception {
     	NodeIterator vi = G.nodeIterator();
		while (vi.hasNext()) {
           	int u = vi.next();
           	if (color[u] == 0) {
               	DFS(u);
              }
     	}
     	System.out.println("fas = " + fas);
	}
	
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		
		//args = new String[] {"cnr-2000"};
		//args = new String[] {"wordassociation-2011"};
		
		if(args.length != 1) {
			System.out.println("Usage: java dfsFAS basename");
			System.out.println("Output: FAS statistics");
			return;
		}
		
		System.out.println("Starting " + args[0]);
		
		dfsFAS fas = new dfsFAS(args[0]);
		fas.computeFAS();
				
		long estimatedTime = System.currentTimeMillis() - startTime;
          System.out.println(args[0] + ": Time elapsed = " + estimatedTime/1000.0);
	}
}
