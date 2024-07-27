// linked list node helper data type
public class Node {
    public Integer item;
    public Node next;
    public Node prev;
    
    public Node() {
      item = 0;
      next = null;
      prev = null;
    }
    
    public Node(Integer i) {
      item = i;
      next = null;
      prev = null;
    }
}