# multiAgents.py
# --------------
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
import time

from multiagent import search
from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        oldFood = currentGameState.getFood()
        food_list = oldFood.asList()
        food_dist = []
        for i in range(len(food_list)):
            food_dist.append((manhattanDistance(newPos, food_list[i])))

        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghostPos = [ghostState.getPosition() for ghostState in newGhostStates]

        dist_to_ghost = []
        for pos in ghostPos:
            dist_to_ghost.append(manhattanDistance(pos, newPos))



        "*** YOUR CODE HERE ***"
        return 1/(min(food_dist)+0.1) - 2 * (1/(min(dist_to_ghost) + 0.0001))


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):

        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, -1, 0)[1]

    def value(self, state, counter, agentIndex):

        agentIndex = agentIndex % state.getNumAgents()
        if agentIndex == 0:
            counter += 1

        if counter >= self.depth or state.isWin() or state.isLose():
            return [self.evaluationFunction(state)]

        if agentIndex == 0:
            return self.max_value(state, counter, agentIndex)
        else:
            return self.min_value(state, counter, agentIndex)

    def max_value(self, state, counter, agentIndex):

        legalMoves = state.getLegalActions(agentIndex)

        maxVal = -float("inf")
        bestMove = None
        for move in legalMoves:
            newState = state.generateSuccessor(agentIndex, move)
            val = self.value(newState, counter, agentIndex + 1)
            if val[0] >= maxVal:
                maxVal = val[0]
                bestMove = move
        return [maxVal, bestMove]

    def min_value(self, state, counter, agentIndex):

        legalMoves = state.getLegalActions(agentIndex)

        minVal = float("inf")
        bestMove = None

        for move in legalMoves:
            newState = state.generateSuccessor(agentIndex, move)
            val = self.value(newState, counter, agentIndex + 1)
            if val[0] <= minVal:
                minVal = val[0]
                bestMove = move

        return [minVal, bestMove]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, -1, 0, -float("inf"), float("inf"))[1]

    def value(self, state, counter, agentIndex, a, b):

        agentIndex = agentIndex % state.getNumAgents()
        if agentIndex == 0:
            counter += 1

        if counter >= self.depth or state.isWin() or state.isLose():
            return [self.evaluationFunction(state)]

        if agentIndex == 0:
            return self.max_value(state, counter, agentIndex, a, b)
        else:
            return self.min_value(state, counter, agentIndex, a, b)

    def max_value(self, state, counter, agentIndex, a, b):

        legalMoves = state.getLegalActions(agentIndex)

        maxVal = -float("inf")
        bestMove = None
        for move in legalMoves:
            newState = state.generateSuccessor(agentIndex, move)
            val = self.value(newState, counter, agentIndex + 1, a, b)
            if val[0] >= maxVal:
                bestMove = move
                maxVal = val[0]
            if maxVal > b:
                return [maxVal]
            a = max(a, maxVal)

        return [maxVal, bestMove]

    def min_value(self, state, counter, agentIndex, a, b):

        legalMoves = state.getLegalActions(agentIndex)

        minVal = float("inf")
        bestMove = None

        for move in legalMoves:
            newState = state.generateSuccessor(agentIndex, move)
            val = self.value(newState, counter, agentIndex + 1, a, b)
            if val[0] <= minVal:
                minVal = val[0]
                bestMove = move
            if minVal < a:
                return [minVal]
            b = min(b, minVal)

        return [minVal, bestMove]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, -1, 0)[1]

    def value(self, state, counter, agentIndex):

        agentIndex = agentIndex % state.getNumAgents()
        if agentIndex == 0:
            counter += 1

        if counter >= self.depth or state.isWin() or state.isLose():
            return [self.evaluationFunction(state)]

        if agentIndex == 0:
            return self.max_value(state, counter, agentIndex)
        else:
            return self.exp_value(state, counter, agentIndex)

    def max_value(self, state, counter, agentIndex):

        legalMoves = state.getLegalActions(agentIndex)

        maxVal = -float("inf")
        bestMove = None
        for move in legalMoves:
            newState = state.generateSuccessor(agentIndex, move)
            val = self.value(newState, counter, agentIndex + 1)
            if val[0] >= maxVal:
                maxVal = val[0]
                bestMove = move
        return [maxVal, bestMove]

    def exp_value(self, state, counter, agentIndex):

        legalMoves = state.getLegalActions(agentIndex)

        expVal = 0

        for move in legalMoves:
            newState = state.generateSuccessor(agentIndex, move)
            val = self.value(newState, counter, agentIndex + 1)
            expVal += val[0]

        expVal = expVal / len(legalMoves)

        return [expVal]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pac_pos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    ghostPos = [ghostState.getPosition() for ghostState in newGhostStates]
    ghostPos_int = []
    for p in ghostPos:
        pos = []
        for p_ in p:
            pos.append(int(p_))

        ghostPos_int.append(tuple(pos))

    dist_to_ghost = []
    for pos in ghostPos_int:
        dist_to_ghost.append(1 / ((mazeDistance(pos, pac_pos, currentGameState) + 0.0001) ** 2))

    food_list = currentGameState.getFood().asList()
    food_dist = []
    for i in range(len(food_list)):
        food_dist.append((mazeDistance(pac_pos, food_list[i], currentGameState)))

    caps = currentGameState.getCapsules()
    capsval = 0
    if caps:
        caps_dist = []
        for c in caps:
            caps_dist.append((mazeDistance(pac_pos, c, currentGameState)))
        capsval = sum(caps_dist) * 20

    ghostval= sum(dist_to_ghost) * 10
    foodval = 0
    if food_dist:
        foodval=min(food_dist)
    scoreval=currentGameState.getScore()
    val = -ghostval - foodval + scoreval - capsval
    return val

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))

# Abbreviation
better = betterEvaluationFunction
