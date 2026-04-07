// Linked List helpers
// Provides ListNode struct and conversion functions for linked list problems

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    fn new(val: i32) -> Self {
        ListNode { val, next: None }
    }
}

/// Convert a Vec<i32> into a linked list
/// Example: vec![1, 2, 3] → 1 -> 2 -> 3 -> None
pub fn list_from_vec(vals: Vec<i32>) -> Option<Box<ListNode>> {
    if vals.is_empty() {
        return None;
    }
    
    let mut head = Box::new(ListNode::new(vals[0]));
    let mut current = &mut head;
    
    for &val in &vals[1..] {
        current.next = Some(Box::new(ListNode::new(val)));
        current = current.next.as_mut().unwrap();
    }
    
    Some(head)
}

/// Convert a linked list back into Vec<i32>
/// Example: 1 -> 2 -> 3 -> None → vec![1, 2, 3]
pub fn list_to_vec(node: Option<Box<ListNode>>) -> Vec<i32> {
    let mut result = Vec::new();
    let mut current = node;
    
    while let Some(node) = current {
        result.push(node.val);
        current = node.next;
    }
    
    result
}

/// Reverse a linked list
/// Useful for some linked list problems
pub fn reverse_list(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    let mut prev = None;
    let mut current = head;
    
    while let Some(mut node) = current {
        let next = node.next.take();
        node.next = prev;
        prev = Some(node);
        current = next;
    }
    
    prev
}
