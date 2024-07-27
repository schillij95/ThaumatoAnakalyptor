import java.io.IOException;
import java.util.BitSet;
import java.util.List;
import java.util.LinkedList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import it.unimi.dsi.logging.ProgressLogger;
import it.unimi.dsi.webgraph.BVGraph;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.IncrementalImmutableSequentialGraph;
import it.unimi.dsi.webgraph.NodeIterator;
import it.unimi.dsi.webgraph.Transform;

public class dllFAS {
	String basename;
	ImmutableGraph G, I;
	int n;
	int numClasses;
	DoublyLinkedList[] bins;
	Node[] nodes;
	int[] deltas;
	int max_delta = Integer.MIN_VALUE;
	List<Integer> seq = null;
		
	public dllFAS(String basename) throws Exception {
		this.basename = basename;
		
		System.out.println("Loading graph...");
		G = ImmutableGraph.load(basename); //We need random access
		System.out.println("Graph loaded");
		
		System.out.println("Transposing graph...");
		I = Transform.transpose(G);
		System.out.println("Graph transposed");
		
		n = G.numNodes();
		System.out.println("n="+n);
		System.out.println("e="+G.numArcs());
		numClasses = 2*n - 3;
		deltas = new int[n];
		nodes = new Node[n];
		bins = new DoublyLinkedList[numClasses];
		createbins();
	}
	
	void createbins() {
		NodeIterator vi = G.nodeIterator();
		while (vi.hasNext()) {
           	int u = vi.next();
           	
           	nodes[u] = new Node(new Integer(u));
        	
           	//int odeg = G.outdegree(u);
           	//int ideg = I.outdegree(u);
           	int odeg = deg(G,u);
           	int ideg = deg(I,u);
        	
           	if(odeg == 0) {
            		addToBin(2-n, nodes[u]);
            		deltas[u] = 2 - n;
           	} else if (ideg == 0 && odeg > 0) {
          		addToBin(n-2, nodes[u]);
          		deltas[u] = n - 2;
           	} else {
               	int d = odeg - ideg;
          		addToBin(d, nodes[u]);
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
		
	void computeseq() {
		List<Integer> s1 = new LinkedList<Integer>();
		List<Integer> s2 = new LinkedList<Integer>();
		
		int numdel = 0;
		while(numdel < n) {
			if (bins[0] != null) {
     			while(!bins[0].isEmpty()) {
      				Integer u = bins[0].remove().item;
      				numdel++;
      				s2.add(0, u);
				
      				deleteNode(u);
     			}
     		}
			
			if (bins[numClasses-1] != null) {
     			while(!bins[numClasses-1].isEmpty()) {
      				Integer u = bins[numClasses-1].remove().item;
      				numdel++;
      				s1.add(u);
				
      				deleteNode(u);
     			}
     		}
						
			if(numdel < n) {
     			if (bins[max_delta - (2 - n)].isEmpty()) {
           			System.out.println("max_delta bin is empty: "+max_delta);
     			}
     			Integer u = bins[max_delta - (2 - n)].remove().item;
     			updateMaxDelta(max_delta);
          		numdel++;
     			s1.add(u);
			
     			deleteNode(u);
           	}
			
		}
		
		s1.addAll(s2);
		seq = s1;
	}
	
	void deleteNode(Integer u) {
  		deltas[u] = Integer.MIN_VALUE;
    		nodes[u] = null;
  		
  		deleteNode(G, u, true);
  		deleteNode(I, u, false);
	}
	
	void deleteNode(ImmutableGraph G, Integer u, boolean out) {
		int[] u_neighbors = G.successorArray(u);
		int u_deg = G.outdegree(u);
		for(int i=0; i<u_deg; i++) {
			int v = u_neighbors[i];
			
			if(v==u)
				continue;
			
			if (nodes[v] != null) {
     			int oldDelta = deltas[v];
     			int newDelta = oldDelta;
			
     			if (out) {
           			newDelta++;
     			} else {
           			newDelta--;
                 	}
           	
               	deltas[v] = newDelta;
               	
               	Node vNode = nodes[v];
               	vNode.prev.next = vNode.next;
               	vNode.next.prev = vNode.prev;
               	addToBin(newDelta, vNode);
               	updateMaxDelta(oldDelta);
           	}
		}
	}
	
	void addToBin(int delta, Node v) {
     	if (bins[delta - (2 - n)] == null)
           	bins[delta - (2 - n)] = new DoublyLinkedList();
	
		bins[delta - (2 - n)].add(v);
		
		//max_delta, min_delta
		if(delta<n-2  && max_delta<delta)
			max_delta=delta;
	}
	
	void updateMaxDelta(int delta) {
		//max_delta
		if(delta==max_delta && bins[delta - (2 - n)].isEmpty()) {
			while(bins[max_delta - (2 - n)].isEmpty()) {
				do {
					max_delta--;
					
					if(max_delta == (2 - n))
     					break;
				}
				while(bins[max_delta - (2 - n)] == null);
				
				if(max_delta == (2 - n))
     					break;
			}
		}
	}
	
	public void createDAG() throws Exception {
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
	}
	
	
	public static void main(String[] args) throws Exception {
		long startTime = System.currentTimeMillis();
		
		//args = new String[] {"wordassociation-2011"};
		//args = new String[] {"cnr-2000"};
		
		if(args.length != 1) {
			System.out.println("Usage: java FVS5 basename");
			System.out.println("Output: dag");
			return;
		}
		
		System.out.println("Starting " + args[0]);
		
		dllFAS fas = new dllFAS(args[0]);
		fas.createDAG();
				
		long estimatedTime = System.currentTimeMillis() - startTime;
          System.out.println(args[0] + ": Time elapsed = " + estimatedTime/1000.0);
	}
}
