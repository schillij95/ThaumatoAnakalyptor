/*
    Array Based FAS Implementation

    Copyright (C) 2016 by
    Michael Simpson <simpsonm@uvic.ca>
    All rights reserved.
    BSD license.
*/

import java.io.IOException;
import java.io.PrintWriter;
import java.util.BitSet;
import java.util.List;
import java.util.LinkedList;

import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;

public class ArrayFAS {

	String basename; // input graph basename
   	ImmutableGraph G, I;// graph G and its inverse I
	int n; // number of vertices in G
	int numClasses; // number of vertex classes
	int[] bins; // holds the tail of bin i
	int[] prev; // holds reference to previous node for vertex i
	int[] next; // holds reference to next node for vertex i
	int[] deltas; // delta class for a vertex. NOTE: determines if a node is present
	int max_delta = Integer.MIN_VALUE;
	List<Integer> seq = null;
		
	// load graph and initialize class variables
	public ArrayFAS(String basename) throws Exception {
		this.basename = basename;
		
		System.out.println("Loading graph...");
		G = ImmutableGraph.load(basename); //We need random access
		System.out.println("Graph loaded");
		
		//System.out.println("Transposing graph...");
		//I = Transform.transpose(G);
		System.out.println("Loading transpose graph...");
		I = ImmutableGraph.load(basename+"-t");
		System.out.println("Graph loaded");
		
		n = G.numNodes();
		System.out.println("n="+n);
		System.out.println("e="+G.numArcs());
		numClasses = 2*n - 3;
		deltas = new int[n];
		next = new int[n]; // init to -1
		prev = new int[n]; // init to -1
		for (int i = 0; i < n; i++) {
      		next[i] = -1;
      		prev[i] = -1;
		}
		bins = new int[numClasses]; // init to -1
		for (int i = 0; i < numClasses; i++) {
      		bins[i] = -1;
		}
		//System.out.println("Creating bins...");		
		createbins();
		//System.out.println("Bins created");	
	}
	
	// initialize bins
	void createbins() {
		NodeIterator vi = G.nodeIterator();
		while (vi.hasNext()) {
           	int u = vi.next();
        	
           	//int odeg = G.outdegree(u);
           	//int ideg = I.outdegree(u);
           	int odeg = deg(G,u);
           	int ideg = deg(I,u);
        	
           	if(odeg == 0) {
            		addToBin(2-n, u);
            		deltas[u] = 2 - n;
           	} else if (ideg == 0 && odeg > 0) {
          		addToBin(n-2, u);
          		deltas[u] = n - 2;
           	} else {
               	int d = odeg - ideg;
          		addToBin(d, u);
          		deltas[u] = d;
           	}
          }      
	}
	
	int deg(ImmutableGraph G, int u) {
		//count degree without self-loops
		int ret = 0;
		int[] u_neighbors = G.successorArray(u);
		int u_deg = G.outdegree(u);
		for(int i=0; i<u_deg; i++) {
			int v = u_neighbors[i];
			if(v==u)
				continue;
			ret++;
		}
		return ret;
	}
	
	// main loop for computing vertex sequence
	void computeseq() throws Exception {
     	String outfile = basename + "_maxDeltas.txt";
     	PrintWriter writer = new PrintWriter(outfile, "UTF-8");
	
		List<Integer> s1 = new LinkedList<Integer>();
		List<Integer> s2 = new LinkedList<Integer>();
		
		int numdel = 0;
		while(numdel < n) {
     		while(bins[0] != -1) {
      			int u = bins[0];
      			bins[0] = prev[u];
      			if(prev[u] != -1)
           			next[prev[u]] = -1;
      			deleteNode(u);
      			
      			numdel++;
      			s2.add(0, u);
     		}
			
     		while(bins[numClasses-1] != -1) {
      			int u = bins[numClasses-1];
      			bins[numClasses-1] = prev[u];
      			if(prev[u] != -1)
           			next[prev[u]] = -1;
      			deleteNode(u);
      			
      			numdel++;
      			s1.add(u);
     		}
						
			if(numdel < n) {
     			if(bins[max_delta - (2 - n)] == -1)
           			System.out.println("max_delta bin is empty: " + max_delta);
     			int u = bins[max_delta - (2 - n)];
     			writer.println(max_delta);
     			bins[max_delta - (2 - n)] = prev[u];
     			if(prev[u] != -1)
           			next[prev[u]] = -1;
           		updateMaxDelta(max_delta);
     			deleteNode(u);
          		
          		numdel++;
     			s1.add(u);
           	}
			
		}
		
		s1.addAll(s2);
		seq = s1;
		writer.close();
	}
	
	// delete a vertex from G
	void deleteNode(int u) {
  		deltas[u] = Integer.MIN_VALUE;
  		
  		deleteNode(G, u, true);
  		deleteNode(I, u, false);
  		
  		prev[u] = -1;
  		next[u] = -1;
	}
	
	// delete a vertex from G by updating the vertex class and bin of its neighbours in G
	void deleteNode(ImmutableGraph G, int u, boolean out) {
		int[] u_neighbors = G.successorArray(u);
		int u_deg = G.outdegree(u);
		for(int i = 0; i < u_deg; i++) {
			int v = u_neighbors[i];
			
			if(v == u)
				continue;
			
			if (deltas[v] > Integer.MIN_VALUE) {
     			int oldDelta = deltas[v];
     			int newDelta = oldDelta;
			
     			if (out) {
           			newDelta++;
     			} else {
           			newDelta--;
                 	}
           	
               	deltas[v] = newDelta;
               	
               	if (bins[oldDelta - (2 - n)] == v)
                   	bins[oldDelta - (2 - n)] = prev[v];
               	
                  if (prev[v] != -1)
                   	next[prev[v]] = next[v];
                  if (next[v] != -1)
                   	prev[next[v]] = prev[v];
               	
               	addToBin(newDelta, v);
               	updateMaxDelta(oldDelta);
           	}
		}
	}
	
    // add vertex v to bin corresponding to delta
    void addToBin(int delta, int v) {
      if (bins[delta - (2 - n)] == -1) {
        bins[delta - (2 - n)] = v;
        prev[v] = -1;
      } else {
        next[bins[delta - (2 - n)]] = v;
        prev[v] = bins[delta - (2 - n)];
        bins[delta - (2 - n)] = v;
      }
      
      next[v] = -1;
		
      if(delta < n-2 && max_delta < delta)
        max_delta = delta;
	}
	
	// update the max delta value
	void updateMaxDelta(int delta) {
		if(delta == max_delta && bins[delta - (2 - n)] == -1) {
			while(bins[max_delta - (2 - n)] == -1) {
     			max_delta--;
					
				if(max_delta == (2 - n))
           			break;
			}
		}
	}

	// Save the computed vertex sequence to a file
	void saveSequenceToFile(String filename) throws IOException {
		PrintWriter writer = new PrintWriter(filename, "UTF-8");
		for (Integer vertex : seq) {
			writer.println(vertex);
		}
		writer.close();
		System.out.println("Vertex sequence saved to " + filename);
	}
	
	// create the DAG from the computed vertex sequence
	public void computeFAS() throws Exception {
		if (seq == null) 
			this.computeseq();
		
		int[] varray = new int[n];
		int i = 0;
		for(Integer u : seq) {
			varray[u] = i;
			i++;
		}
		
		BitSet fvs = new BitSet(n);
		int fas = 0;
		int self = 0;
        
        NodeIterator vi = G.nodeIterator();
        while (vi.hasNext()) {
			int v = vi.next();
			
			int[] v_neighbors = G.successorArray(v);
			int v_deg = G.outdegree(v);
			
			for(int x = 0; x < v_deg; x++) { 
				int w = v_neighbors[x];
				
				if(v==w) { // Self-loop, ignore
      				self++;
					continue;
				}
				
				if (varray[v] > varray[w]) {
					fvs.set(v);
					fas++;
				}
      		}
		}
        
        System.out.println("fvs size is " + fvs.cardinality());
        System.out.println("fas size is " + fas);
        //System.out.println("self loops = " + self);

		// Save the sequence and FAS results
    	saveSequenceToFile(basename + "_sequence.txt");
	}
	
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		
		//args = new String[] {"cnr-2000"};
		// args = new String[] {"wordassociation-2011"};
		
		if(args.length != 1) {
			System.out.println("Usage: java ArrayFAS basename");
			System.out.println("Output: FAS statistics");
			return;
		}
		
		System.out.println("Starting " + args[0]);
		
		ArrayFAS fas = new ArrayFAS(args[0]);
		fas.computeFAS();
				
		long estimatedTime = System.currentTimeMillis() - startTime;
          System.out.println(args[0] + ": Time elapsed = " + estimatedTime/1000.0);
	}
}
// javac -cp "lib/*" -d bin *.java