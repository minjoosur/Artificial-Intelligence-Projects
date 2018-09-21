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


from util import manhattanDistance
from game import Directions
import random, util
import sys

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
        legalMoves = gameState.getLegalActions() # ['North', 'South' , ..]

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        newPos = successorGameState.getPacmanPosition() #(x,y)
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        currPos = currentGameState.getPacmanPosition()


        "*** YOUR CODE HERE ***"
        # if there is probability to be eaten if it moves
        for ghostState in newGhostStates: #ghostState = 
            if ghostState.scaredTimer == 0 and ghostState.getPosition()==newPos:
                return -1
        if action == 'Stop':
            return -1 if ghostState.scaredTimer else 0
        cost = []
        for food in currentGameState.getFood().asList():      
            if not manhattanDistance(newPos,food):
                return 2
            cost.append(manhattanDistance(newPos,food))
        count = 1/min(cost)
        # print(action, count)
        return count

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def get_value(self, state, depth, agents):
        
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agents == state.getNumAgents():
            agents = 0
            depth -= 1

        if depth == 0:
            return self.evaluationFunction(state)

        if agents == 0:
            v = self.max_value(state, depth, agents)
        else:
            v = self.min_value(state, depth, agents)

        return v

    def max_value(self, state, depth, agents):
        v = -float("inf")

        for action in state.getLegalActions(agents):
            v = max(self.get_value(state.generateSuccessor(agents, action), depth, agents + 1), v)

        return v

    def min_value(self, state, depth, agents):
        v = float("inf")

        for action in state.getLegalActions(agents):
            v = min(self.get_value(state.generateSuccessor(agents, action), depth, agents + 1), v)

        return v


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
        
        best_action = None
        best_val = -float("inf")
        

        for action in gameState.getLegalActions():
            
            val = self.get_value(gameState.generateSuccessor(0, action), self.depth, 1)

            if (val >= best_val):
                best_val = val
                best_action = action

        return best_action
    # def getAction(self, gameState):
    #     """
    #     Returns the minimax action from the current gameState using self.depth
    #     and self.evaluationFunction.

    #     Here are some method calls that might be useful when implementing minimax.

    #     gameState.getLegalActions(agentIndex):
    #     Returns a list of legal actions for an agent
    #     agentIndex=0 means Pacman, ghosts are >= 1

    #     gameState.generateSuccessor(agentIndex, action):
    #     Returns the successor game state after an agent takes an action

    #     gameState.getNumAgents():
    #     Returns the total number of agents in the game

    #     gameState.isWin():
    #     Returns whether or not the game state is a winning state

    #     gameState.isLose():
    #     Returns whether or not the game state is a losing state
    #     """
    #     "*** YOUR CODE HERE ***"

    #     def value(state, agtIdx, currDepth):
    #         #if terminal value, return evaluation func
    #         if currDepth == self.depth and agtIdx % state.getNumAgents() == 0:
    #             return (self.evaluationFunction(state), None)
    #         #if agent is pacman, calculate the best action
    #         if agtIdx == 0:
    #             return maxValue(state, agtIdx, currDepth)
    #         #if agent is a ghost, calcaulte the worst action
    #         return minValue(state,agtIdx,currDepth)

    #     def maxValue(state, agtIdx, currDepth):

    #         # base case, game over
    #         if state.isWin() or state.isLose():
    #             return (self.evaluationFunction(state), None)
            
    #         bestValue = float("-inf")
    #         bestAction = None

    #         for currAction in state.getLegalActions(agtIdx):
    #             nextState = gameState.generateSuccessor(agtIdx, currAction)
    #             currValue = value(nextState, agtIdx+1, currDepth)

    #             if currValue[0] > bestValue:
    #                 bestValue = currValue[0]
    #                 bestAction = currAction
            
    #         return (bestValue, bestAction)

    #     def minValue(state, agtIdx, currDepth):
    #         legalActions = state.getLegalActions(agtIdx)
    #         # base case, game over
    #         if state.isWin() or state.isLose() or not legalActions:
    #             return (self.evaluationFunction(state), None)

    #         worstValue = float("inf")
    #         worstAction = None

    #         for currAction in legalActions:
    #             nextState = gameState.generateSuccessor(agtIdx, currAction)

    #             #if last ghost, go deeper depth
    #             if agtIdx == state.getNumAgents() - 1:
    #                 currValue = value(nextState, 0, currDepth+1)
    #             #if not last ghost, stay in this depth and check other ghosts
    #             else:
    #                 currValue = value(nextState, agtIdx+1, currDepth)
                
    #             if currValue[0] < worstValue:
    #                 worstValue = currValue[0]
    #                 worstAction = currAction

    #         return (worstValue, worstAction)
    #     return value(gameState, self.index, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

def get_value(self, state, depth, agents, alpha, beta):
        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agents == state.getNumAgents():
            agents = 0
            depth -= 1

        if depth == 0:
            return self.evaluationFunction(state)

        if agents == 0:
            v = self.max_value(state, depth, agents, alpha, beta)
        else:
            v = self.min_value(state, depth, agents, alpha, beta)

        return v

    def max_value(self, state, depth, agents, alpha, beta):
        v = -float("inf")

        for action in state.getLegalActions(agents):
            v = max(self.get_value(state.generateSuccessor(agents, action), depth, agents + 1, alpha, beta), v)
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, state, depth, agents, alpha, beta):
        v = float("inf")

        for action in state.getLegalActions(agents):
            v = min(self.get_value(state.generateSuccessor(agents, action), depth, agents + 1, alpha, beta), v)
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        best_val = -float("inf")

        alpha = -float("inf")
        beta = float("inf")

        for action in gameState.getLegalActions():
            val = self.get_value(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta)
            if val >= best_val:
                best_val = val
                best_action = action
            alpha = max(best_val, alpha)

        return best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_value(self, state, depth, agents):

        if state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if agents == state.getNumAgents():
            agents = 0
            depth -= 1

        if depth == 0:
            return self.evaluationFunction(state)

        if agents == 0:
            v = self.max_value(state, depth, agents)
        else:
            v = self.exp_value(state, depth, agents)

        return v

    def max_value(self, state, depth, agents):
        v = -float("inf")

        for action in state.getLegalActions(agents):
            v = max(self.get_value(state.generateSuccessor(agents, action), depth, agents + 1), v)

        return v

    def exp_value(self, state, depth, agents):
        v = 0
        for action in state.getLegalActions(agents):
            p = 1.0 / len(state.getLegalActions(agents))
            v += p * self.get_value(state.generateSuccessor(agents, action), depth, agents + 1)

        return v

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
        # Immediately updated in get_value
        best_action = None
        best_val = -float("inf")
        # print("index: " + str(self.index))

        for action in gameState.getLegalActions():
            # print("action: " + str(action))
            val = self.get_value(gameState.generateSuccessor(0, action), self.depth, 1)

            if val >= best_val:
                best_val = val
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()

    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodCountFeature = 1 / (1.0 + food.count())
    minDistToFoodFeature = 1 / (1.0 + min(
        [util.manhattanDistance(foodPos, currentPos) if food[foodPos[0]][foodPos[1]] and food.count() > 0 else 0 for
         foodPos in food.asList()] or [0]))
    minDistToGhostFeature = min([util.manhattanDistance(ghost.getPosition(), currentPos) for ghost in ghostStates])
    averageDistToGhostFeature = sum([util.manhattanDistance(ghost.getPosition(), currentPos) for ghost in ghostStates]) / (len(ghostStates))

    scoreFeature = currentGameState.getScore()
    averageScaredTimeFeature = (sum(scaredTimes) / len(scaredTimes))
    minScaredTimeFeature = min(scaredTimes)


    if minDistToGhostFeature <= 2:
        minDistToGhostWeight = -200
        scoreWeight = 0.01
    else:
        scoreWeight = 100
        minDistToGhostWeight = 0

    averageScaredTimeWeight = 0
    foodCountWeight = 0
    minDistToFoodWeight = 10
    minScaredTimeWeight = 100
    averageDistToGhostWeight = 0

    score = minDistToGhostWeight * (minDistToGhostFeature) \
            + scoreWeight * (scoreFeature) \
            + averageScaredTimeWeight * (averageScaredTimeFeature) \
            + foodCountWeight * (foodCountFeature) \
            + minDistToFoodWeight * (minDistToFoodFeature) \
            + minScaredTimeWeight * (minScaredTimeFeature) \
            + averageDistToGhostWeight * (averageDistToGhostFeature)

    return score

# Abbreviation
better = betterEvaluationFunction
