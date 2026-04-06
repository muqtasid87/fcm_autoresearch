import numpy as np
import itertools
#import gmsh
import matplotlib.pyplot as plt
from fcm_quadrature.data_generation.parameters import ProjectParameter

class Point:
    def __init__(self, vec, idxEdge=np.empty):
        '''vec: 2D vector, coordinates of the point, numpy array'''
        self.vec = np.array(vec)
        self.idxEdge = idxEdge
        # self.r = None
        # self.phi = None

    def getX(self):
        return self.vec[0]

    def getY(self):
        return self.vec[1]

    def getDistance(self, secondPoint):
        '''get distance to another point'''
        p2pVec = self.vec - secondPoint.vec
        return np.linalg.norm(p2pVec, ord=2)

    # def getClosestPoint(self, pointList):
    #     allDistance = []
    #     for point in pointList:
    #         allDistance.append(self.getDistance(point))
    #     idxClosest = np.argmin(np.array(allDistance))
    #     return pointList[idxClosest], idxClosest

    def getClosestPointOnVector(self, pointList, vec):
        minDotProduct = np.inf
        for point in pointList:
            dotProduct = np.dot(point.vec - self.vec, vec)
            if dotProduct > 0 and dotProduct < minDotProduct:
                minDotProduct = dotProduct
                closestPoint = point
        return closestPoint

    def isOverlap(self, secondPoint):
        return np.allclose(self.vec, secondPoint.vec)

    def getMaxDeviation(self, directionUnitVec, cell, maxDevLength):
        '''get the max deviation at current point (distance to the boundary of the cell)
        along the direction vector defined by directionUnitVec. The returned value is
        capped at maxDevLength.
        '''
        deviationLine = Line(self, Point(self.vec + directionUnitVec))
        intersectionPoint, _ = cell.getIntersectionPoints(deviationLine)
        devLength = np.linalg.norm(intersectionPoint.vec - self.vec, ord=2)
        maxDev = min(devLength, maxDevLength)
        return maxDev

    # def translateToNewCoord(self, newOrigVec):
    #     newPointVec = self.vec - newOrigVec
    #     return Point(newPointVec)

    # def transToPolar(self):
    #     '''transform the point's Cartesian coordinates into polar coordinates'''
    #     x = self.getX()
    #     y = self.getY()
    #     self.r = np.sqrt(x**2 + y**2)
    #     self.phi = np.arctan2(y, x)

    def plot(self, ax, label='', color='blue'):
        ax.plot(self.getX(), self.getY(), '.', label=label, color=color)

    def annotate(self, ax, text, xytext=(5, 5), size=10):
        ax.annotate(text,
            xy=(self.getX(), self.getY()),
            xycoords='data',
            xytext=xytext,
            textcoords='offset points',
            size=size,
        )

class PointList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(args)

    def __add__(self, anotherPointList):
        newList = PointList(*self)
        for point in anotherPointList:
            newList.append(point)
        return newList

    def removeDuplicates(self):
        foundDuplicates = False
        for pointA, pointB in itertools.combinations(self, 2):
            if pointA.isOverlap(pointB):
                self.remove(pointA)
                foundDuplicates = True
        return foundDuplicates

    def removeOnLinePoints(self, line):
        for point in self:
            if line.isOnLine(point):
                self.remove(point)

    def print(self):
        for idx, point in enumerate(self):
            print(f'point {idx}: {point.vec}')

    def getCoordinatesList(self):
        coordinates = [
            self[2].vec.tolist(),
            self[3].vec.tolist(),
            self[1].vec.tolist(),
            self[0].vec.tolist(),
        ]
        # for point in self:
        #     coordinates.append(point.vec.tolist())
        return coordinates

    def plot(self, ax, label='', color='blue', **kwargs):
        ax.plot(
            [point.getX() for point in self],
            [point.getY() for point in self],
            '.',
            label=label,
            color=color,
            **kwargs,
        )

    def showIdx(self, ax):
        for idx, point in enumerate(self):
            ax.annotate(
                str(idx),
                xy=(point.getX(), point.getY()),
                xycoords='data',
                xytext=(5, 5),
                textcoords='offset points',
            )

    def toClosedLineList(self):
        lines = LineList()
        for idx, pointA in enumerate(self[0:-1]):
            pointB = self[idx + 1]
            lines.append(Line(pointA, pointB))
        lines.append(Line(self[-1], self[0]))
        return lines

    def plotAsLines(self, ax, label='', color='green'):
        lines = self.toClosedLineList()
        lines.plot(ax, label=label, color=color)

class Polygon:
    def __init__(self, vertices: PointList):
        self.allVertices = vertices
        self.allEdges = vertices.toClosedLineList()

    def plotOutline(self, ax, label='', color='orange', **kwargs):
        self.allEdges.plot(ax, label=label, color=color, **kwargs)

    def fill(self, ax, label='', color='green', **kwargs):
        ax.fill(
            [point.getX() for point in self.allVertices],
            [point.getY() for point in self.allVertices],
            label=label,
            color=color,
            **kwargs,
        )

class Line:
    # point startPoint
    # point endPoint
    # length: length of the line section
    # vec: vector representation of the line.
    #   The equation of a line is ax+by+c=0, while vec = [a, b, c]

    def __init__(self, startPoint, endPoint, idx=np.empty):
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.length = startPoint.getDistance(endPoint)
        self.sectionVec = endPoint.vec - startPoint.vec
        startPointVec = np.concatenate((startPoint.vec, [1]))
        endPointVec = np.concatenate((endPoint.vec, [1]))
        self.vec = np.cross(startPointVec, endPointVec)
        self.idx = idx

    def getIntersection(self, secondLine):
        ''' get the coordinates of the intersection point with another line
        return False if 2 lines are parallel
        the Point will have the index of the current object as the edge index
        '''
        x, y, z = np.cross(self.vec, secondLine.vec)        
        if z == 0:
            return False
        else:
            return Point([x/z, y/z], idxEdge=self.idx)

    def getMidPoint(self):
        startVec = self.startPoint.vec
        endVec = self.endPoint.vec
        midPoint = Point((startVec + endVec)/2)
        return midPoint

    def getOnLinePoints(self, midPointRatios):
        midPoints = PointList()
        for ratio in midPointRatios:
            startVec = self.startPoint.vec
            endVec = self.endPoint.vec
            midPointVec = (endVec - startVec)*ratio + startVec
            midPoints.append(Point(midPointVec))
        return midPoints

    def getDistanceToPoint(self, point: Point):
        x = point.vec[0]
        y = point.vec[1]
        a = self.vec[0]
        b = self.vec[1]
        c = self.vec[2]
        distance = np.abs(a*x + b*y + c) / np.sqrt(a*a + b*b)
        return distance

    def getSignedDistanceToPoint(self, point: Point):
        '''
        Returns signed distance from point to line.
        Positive if point is on the side where the normal (a, b) points,
        negative otherwise.
        '''
        x = point.vec[0]
        y = point.vec[1]
        a = self.vec[0]
        b = self.vec[1]
        c = self.vec[2]
        signed_distance = (a*x + b*y + c) / np.sqrt(a*a + b*b)
        return signed_distance

    # def getPointsFromLine(self, numPoint, method, startRatio=0.01):
    #     endRatio = 1 - startRatio
    #     if method == 'linear':
    #         allRatio = np.linspace(startRatio, endRatio, numPoint)
    #     elif method == 'log':
    #         halfArray = np.logspace(
    #             np.log10(startRatio),
    #             np.log10(endRatio/2),
    #             int(numPoint/2) + 1,
    #         )
    #         allRatio = np.concatenate((halfArray, np.flip(1 - halfArray[:-1])))

    #     startVec = self.startPoint.vec
    #     endVec = self.endPoint.vec
    #     deltaVec = endVec - startVec
    #     allPoints = PointList()
    #     for ratio in allRatio:
    #         newVec = startVec + ratio * deltaVec
    #         allPoints.append(Point(newVec))
    #     return allPoints



    def isOnLine(self, point):
        pointVec = np.concatenate((point.vec, [1]))
        t = np.dot(self.vec, pointVec)
        if np.abs(t) < 1e-12:
            return True
        else:
            return False

    def isInLineSection(self, point):
        # check if the given point is within the line section defined
        # by the start and end point
        if self.isOnLine(point) == False:
            return False
        distanceToStart = self.startPoint.getDistance(point)
        distanceToEnd = self.endPoint.getDistance(point)
        if distanceToStart + distanceToEnd == self.length:
            return True
        else:
            return False

    def getInwardUnitVec(self):
        '''get the unit vector pointing inward to the mesh wrt the cut line
        return a vector as 2D numpy array
        inward is defined as the right hand side along the line vector direction
        '''
        cutVec = np.concatenate((self.sectionVec, [0]))
        inwardVec = np.cross(cutVec, np.array([0, 0, 1]))
        # keep the x, y coordinates
        inwardVec = inwardVec[0:2]
        # normalize the vector
        self.inwardUnitVec = inwardVec/np.linalg.norm(inwardVec, ord=2)
        return self.inwardUnitVec

    def plot(self, ax, label='', color='orange', **kwargs):
        ax.plot(
            [self.startPoint.getX(), self.endPoint.getX()],
            [self.startPoint.getY(), self.endPoint.getY()],
            # '-',
            label=label,
            color=color,
            **kwargs,
        )

    def plotAsVec(self, ax, size=10):
        ax.annotate(
            '',
            xy=(self.endPoint.getX(), self.endPoint.getY()),
            xycoords='data',
            xytext=(self.startPoint.getX(), self.startPoint.getY()),
            textcoords='data',
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="k"),
            size=size,
        )


class LineList(list):
    def __init__(self, *args, **kwargs):
        super(LineList, self).__init__(args)

    def samplePoints(self, numPoints: int, method: str, startRatio: float):
        '''
        sample points on a list of lines

        Input:
        - numPoints: number of points to sample, on each line
        - method: defines the distribution of point positions, 'linear' or 'log'.
        - startRatio:
        '''
        if method == 'linear':
            allRatio = np.linspace(startRatio, 1 - startRatio, numPoints)
        elif method == 'log':
            if (numPoints%2) == 0:
                halfArray = np.geomspace(startRatio, 0.5 - startRatio, int(numPoints/2))
                allRatio = np.concatenate((halfArray, np.flip(1 - halfArray)))
            else:
                halfArray = np.geomspace(startRatio, 0.5, int((numPoints+ 1)/2))
                allRatio = np.concatenate((halfArray, np.flip(1 - halfArray[:-1])))

        allPoints = PointList()
        for singleLine in self:
            allPoints += singleLine.getOnLinePoints(allRatio)
        #     tmpPointList = edge.getPointsFromLine(
        #         numPoints, 
        #         method, 
        #         startRatio=startRatio,
        #     )
        #     # remove end points that overlap with the end side
        #     for oppositeEdge in allOppositeEdges:
        #         tmpPointList.removeOnLinePoints(oppositeEdge)
        #     allPoints += tmpPointList
        return allPoints

    def plot(self, ax, label='', color='orange', **kwargs):
        for idx, line in enumerate(self):
            line.plot(
                ax,
                label=label if idx == 0 else '',
                color=color,
                **kwargs,
            )


class Arc:
    def __init__(self, startPoint, endPoint, direction, radiusRatio) -> None:
        '''
        r: the radius of the arc, should be larger than chord.length/2
        direction: determin which side the center of the arc is located
            1 for inward (right hand side along the line vector direction)
            -1 for outward (left hand side)
        '''
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.chord = Line(startPoint, endPoint)
        self.r = self.chord.length*radiusRatio
        
        # center point        
        midPoint = self.chord.getMidPoint()
        unitVec = self.chord.getInwardUnitVec()
        dist = np.sqrt(self.r**2 - (self.chord.length/2)**2)
        # the center point is on the opposite side
        self.centerPoint = Point(midPoint.vec - dist*unitVec*direction)
        
        # use the center point as the origin of a polar coordinate system
        # represent the arc in polar coordinates (start and end angles)
        newStartPoint = Point(startPoint.vec - self.centerPoint.vec)
        newEndPoint = Point(endPoint.vec - self.centerPoint.vec)
        self.phiStart = np.arctan2(newStartPoint.vec[1], newStartPoint.vec[0])
        self.phiEnd = np.arctan2(newEndPoint.vec[1], newEndPoint.vec[0])

    def getOnArcPoints(self, ratios):
        '''get points on the arc based on given ratio values
        ratios: 1D numpy array, has values between 0 and 1.
        '''
        allPhi = self.phiStart + (self.phiEnd - self.phiStart)*ratios
        allVec = np.array([self.r*np.cos(allPhi), self.r*np.sin(allPhi)]).transpose()
        pointVec = allVec + self.centerPoint.vec
        points = PointList()
        for vec in pointVec:
            points.append(Point(vec))
        return points

class Rectangle:
    # sideLength
    # allVertices: a list of all 4 vertices
    # allSides: a list of all 4 sides (line objects)
    # pointFirstIntersection
    # pointLastIntersection
    # idxFirstIntersection
    # idxLastIntersection

    def __init__(self, sideLengthX, sideLengthY):
        # create the vertices and then sides
        self.sideLengthX = sideLengthX
        self.sidelengthY = sideLengthY
        lx = sideLengthX/2
        ly = sideLengthY/2
        self.allVertices = PointList(
            Point([lx, ly]),
            Point([-lx, ly]),
            Point([-lx, -ly]),
            Point([lx, -ly]),
        )
        self.allEdges = LineList(
            Line(self.allVertices[0], self.allVertices[1], idx=0),
            Line(self.allVertices[1], self.allVertices[2], idx=1),
            Line(self.allVertices[2], self.allVertices[3], idx=2),
            Line(self.allVertices[3], self.allVertices[0], idx=3),
        )
        self.area = sideLengthX*sideLengthY

    def getNextEdgeIdx(self, currentIdx):
        nextIdx = currentIdx + 1
        if nextIdx == 4:
            return 0
        else:
            return nextIdx

    def getIntersectionPoints(self, line):
        '''
        return the intersection points between a line and the square itself.
        Input:
            line: the cutting line, Line object.
        Output:
            pointFirstIntersection: the intersection points along the line vector direction.
            pointLastIntersection: the intersection points along the negative line vector direction.
                  firstIntersection <-- endPoint <-- startPoint <-- lastIntersection
        '''

        allIntersectionPoints = PointList()
        for edge in self.allEdges:
            intersectionPoint = edge.getIntersection(line)
            if intersectionPoint != False:
                if self.isOnSquare(intersectionPoint):
                    allIntersectionPoints.append(intersectionPoint)

        allIntersectionPoints.removeDuplicates()

        numIntersectionPoints = len(allIntersectionPoints)
        if numIntersectionPoints == 0:
            # no intersection was found
            return False
        elif numIntersectionPoints == 1:
            # only one intersection point
            pointFirstIntersection = allIntersectionPoints[0]
            pointLastIntersection = pointFirstIntersection
        elif numIntersectionPoints == 2:
            # 2 intersection points
            vecIntersection = allIntersectionPoints[1].vec - \
                allIntersectionPoints[0].vec
            if np.dot(vecIntersection, line.sectionVec) > 0:
                # the intersection vector and the line vector is in the same direction
                pointFirstIntersection = allIntersectionPoints[1]
                pointLastIntersection = allIntersectionPoints[0]
            else:
                # the intersection vector and the line vector is in the opposite direction
                pointFirstIntersection = allIntersectionPoints[0]
                pointLastIntersection = allIntersectionPoints[1]

        return pointFirstIntersection, pointLastIntersection

    def isOnSquare(self, point):
        for edge in self.allEdges:
            if edge.isInLineSection(point):
                return True
        return False

    def isOutsideSquare(self, points: PointList):
        lx = self.sideLengthX/2
        ly = self.sidelengthY/2
        for point in points:
            x, y = point.vec
            if np.abs(x) > lx or np.abs(y) > ly:
                return True
        return False

class Square(Rectangle):
    def __init__(self, sideLength):
        super().__init__(sideLength, sideLength)


class Mesh:

    def __init__(
        self, 
        par: ProjectParameter, 
        boundary: Square, 
        cell: Square,
        targets: PointList,
    ):
        self.boundary = boundary
        self.cell = cell
        self.targets = targets
        self.par = par
        self.allVertices = PointList()
        self.allMeshPoints = PointList()

    def setCutParameters(self, startPoint, endPoint, direction=None, radiusRatio=None):
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.cutLine = Line(self.startPoint, self.endPoint)
        if self.par.cutSectionType == 'arc':
            self.direction = direction
            self.radiusRatio = radiusRatio 

    def generateMidPoints(self):
        '''
        generate middle points on the cutting section
        '''        
        if self.par.midPointRatios.size == 0:
            self.allMidPoints = PointList()
        else:
            if self.par.cutSectionType == 'line':
                self.allMidPoints = self.cutLine.getOnLinePoints(self.par.midPointRatios)
            elif self.par.cutSectionType == 'arc':
                self.cutArc = Arc(self.startPoint, self.endPoint, 
                    self.direction, self.radiusRatio)
                self.allMidPoints = self.cutArc.getOnArcPoints(self.par.midPointRatios)

    def verifyMidPoints(self):
        '''
        check if all points in self.allMidPoints is within the cell.
        return True if any point is outside of the cell.
        '''
        return self.cell.isOutsideSquare(self.allMidPoints)

    def generateCutConfig(self):
        allVertices = self.getCutVertices(self.boundary)
        self.allMeshPoints = allVertices + self.allMidPoints
        
    def getAreaFraction(self):
        vertices = self.getCutVertices(self.cell)
        x = np.array([point.getX() for point in vertices])
        y = np.array([point.getY() for point in vertices])
        meshArea = 0.5*np.abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
        )
        return meshArea/self.cell.area

    def getCutVertices(self, boundary: Square):
        '''
        calculate the vertices of mesh for the given cut config defined in 'setCutParameters'.
        Output:
        - self.allVertices: a PointList object containing all the vertices, starting with 
            the endPoint of the cutting line.
        '''
        # self.cutLine = Line(self.startPoint, self.endPoint)
        self.allVertices = PointList()
        self.allVertices.append(self.endPoint)
        firstVertex, lastVertex, = boundary.getIntersectionPoints(
            self.cutLine)

        self.allVertices.append(firstVertex)
        idxCurrent = firstVertex.idxEdge
        lastVertexReached = False
        while lastVertexReached == False:
            if idxCurrent == lastVertex.idxEdge:
                # the edge corresponds to the last vertex has been reached
                self.allVertices.append(lastVertex)
                lastVertexReached = True
            else:
                # go to the next edge
                # step the index
                idxCurrent = boundary.getNextEdgeIdx(idxCurrent)
                # add the corresponding boundary vertices
                self.allVertices.append(boundary.allVertices[idxCurrent])

        self.allVertices.append(self.startPoint)
        # check for duplicates (if an intersection overlap with a boundary vertex)
        self.allVertices.removeDuplicates()
        return self.allVertices

    def getCutCoordinates(self):
        '''
        collect all points on the cutLine, then put their coordinates into
        a numpy matrix (n by 2, n is the number of points).
        points order: startPoint -> midPoints -> endPoint
        '''
        allPoints = PointList(
            self.cutLine.startPoint,
            *self.allMidPoints,
            self.cutLine.endPoint,
        )
        cutCoordinates = []
        for point in allPoints:
            cutCoordinates.append([point.getX(), point.getY()])
        return np.array(cutCoordinates)

    def getCutDistances(self):
        '''
        get the distances between the cut line and the four vertices of the cell.
        '''
        allDistance = []
        for targetPoint in self.targets:
            allDistance.append(self.cutLine.getDistanceToPoint(targetPoint))
        return np.array(allDistance)

    def getSignedCutDistances(self):
        '''
        Get signed distances between the cut line and all target points.
        Positive distance means the point is on the retained side of the cut,
        negative means it's on the removed side.
        '''
        allDistance = []
        for targetPoint in self.targets:
            allDistance.append(self.cutLine.getSignedDistanceToPoint(targetPoint))
        return np.array(allDistance)

    def visualize(self, maxEigVal=np.nan):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()
        self.allMeshPoints.plot(ax, label='mesh points')
        self.allMeshPoints.showIdx(ax)
        self.allMeshPoints.plotAsLines(ax, label='mesh', color='green')
        self.cell.allEdges.plot(ax, label='cell')
        self.cutLine.plotAsVec(ax)
        self.targets.plot(ax, label='targets', color='red')
        if not np.isnan(maxEigVal):
            ax.set_title(f'stabilization parameter: {maxEigVal}')
        ax.set_xlim(-2*self.par.scalingFactorX, 2*self.par.scalingFactorY)
        ax.set_ylim(-2*self.par.scalingFactorX, 2*self.par.scalingFactorY)
        ax.legend()
        ax.grid(True)
        # plt.show()
        return fig, ax

    def generateMesh(self, jobID):
        # print(f'cut start point: {self.cutLine.startPoint.vec}, \
        #  end point: {self.cutLine.endPoint.vec}')
        gmsh.initialize()
        gmsh.option.setNumber('General.Verbosity', 1)
        
        gmsh.model.add(self.par.projectName)
        for idxVertex, meshPoint in enumerate(self.allMeshPoints):
            gmsh.model.geo.addPoint(
                meshPoint.getX(),
                meshPoint.getY(),
                0,
                self.par.meshSize*min(self.par.scalingFactorX, self.par.scalingFactorY),
                idxVertex
            )
            # print(f'vertex {idxVertex}: {meshPoint.vec}')
        # add lines
        numMeshPoints = len(self.allMeshPoints)
        for idxLine in range(numMeshPoints - 1):
            gmsh.model.geo.addLine(idxLine, idxLine + 1, idxLine)
        # add the last line
        gmsh.model.geo.addLine(numMeshPoints - 1, 0, numMeshPoints)
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(dim=2)
        filePath = f'{self.par.tempFolderName}/{jobID}.{self.par.meshFileFormat}'

        gmsh.write(filePath)
        gmsh.clear()
        return filePath

