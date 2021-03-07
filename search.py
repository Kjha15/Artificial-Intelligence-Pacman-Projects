# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    call_stack = util.Stack() #call the stack function from util.py
    Start = problem.getStartState()
    start_node = (Start, []) # add an empty list to the stack
    explored = [] # make an empty list 
    call_stack.push(start_node)
    while call_stack.isEmpty() != True:
            running_state, actions = call_stack.pop() # check whether the stack is empty, if not pop out the last item
        
            if running_state not in explored: # if the explored node is not in  the stack, add it
                explored.append(running_state)

                if problem.isGoalState(running_state): # if the explored node or state is the goal state then return all the previous actions and stop the pacman
                    return actions
                else:
                    successor = problem.getSuccessors(running_state) #if not, then look for adjacent (successor) nodes i.e keep searching

                    for upd_state, upd_dir, upd_cost in successor: # this loop will add successor nodes in the stack
                        upd_action = actions + [upd_dir]
                        upd_node = (upd_state, upd_action)
                        call_stack.push(upd_node)
    return actions #this will end the game as pacman will reach the goal state after checking every nodes 


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
     #call the queue function from util.py
    call_queue = util.Queue()
    Start = problem.getStartState()
    # add an empty list to the queue
    start_node = (Start, []) 
    # make an empty list for explored nodes so that we do not check them again
    explored = [] 
    call_queue.push(start_node)
    while call_queue.isEmpty() != True:
            running_state, actions = call_queue.pop() # check whether the queue is empty, if not pop out the last item
        
            if running_state not in explored: # if the explored node is not in the queue, add it  
                explored.append(running_state)
                # if the explored node or state is the goal state then return all the previous actions and stop the pacman
                # I it is not the goal state, receive all values of its adjacent nodes
                if problem.isGoalState(running_state): 
                    return actions
                else:
                    successor = problem.getSuccessors(running_state) 

                    for upd_state, upd_dir, upd_cost in successor: # this loop will add successor nodes in the stack
                        upd_action = actions + [upd_dir]
                        upd_node = (upd_state, upd_action)
                        call_queue.push(upd_node)
    return actions #this will end the game as pacman will reach the goal state after checking every nodes 


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    call_queue = util.PriorityQueue() #priorityqueue for UCS
    explored_state = [] #for adding states which are already explored
    start_state = problem.getStartState()
    
    call_queue.push((start_state, []) ,0) 
    while not call_queue.isEmpty():
        running_states, actions = call_queue.pop()
        if problem.isGoalState(running_states): #return the value if the explored state in running_states is the goal state
                return actions
        if running_states not in explored_state:
            successor = problem.getSuccessors(running_states)

            for updated_successor in successor: #call for each neighbouring successor
                new_cord = updated_successor[0]
                if new_cord not in explored_state:
                    new_dir = updated_successor[1]
                    new_cost = actions + [new_dir]
                    call_queue.push((new_cord, actions + [new_dir]), problem.getCostOfActions(new_cost))
        explored_state.append(running_states)
    return actions


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    call_queue = util.PriorityQueue() #using priorityqueue for A*
    explored_state = []
    start_state = problem.getStartState()
    
    call_queue.push((start_state, []), nullHeuristic(start_state, problem)) #calling huerustics function
    new_cost = 0
    while not call_queue.isEmpty():
        running_states, actions = call_queue.pop()
        if problem.isGoalState(running_states):
                return actions
        if running_states not in explored_state:
            successor = problem.getSuccessors(running_states)

            for updated_successor in successor:
                new_cord = updated_successor[0] #assigning variable for updating the successor
                if new_cord not in explored_state: #update the successor in the list
                    new_dir = updated_successor[1]
                    new_actions = actions + [new_dir]
                    new_cost = problem.getCostOfActions(new_actions) + heuristic(new_cord, problem) #adding the heuristic value 
                    call_queue.push((new_cord, actions + [new_dir]), new_cost)
        explored_state.append(running_states) 
    return actions
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
