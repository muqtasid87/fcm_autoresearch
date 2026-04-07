"""
Modified Job.py for computing quadrature weights instead of eigenvalues.

This version directly calls moment_fitting.py without the integration helper.

Features:
1. Moment fitting weight computation using Green's theorem
2. Optional all-edges mode: start points from all 4 edges (useAllEdgesAsStart=True)
3. Optional polygon vertex saving (savePolygonVertices=True)

CSV Format (depends on savePolygonVertices setting):

If savePolygonVertices=False (default):
   - Columns 0-11: Signed perpendicular distances from cut line to target points
   - Columns 12-15: 4 quadrature weights for basis {1, x, y, xy}
   Total: 16 columns

If savePolygonVertices=True:
   - Columns 0-11: Signed distances
   - Column 12: Number of vertices in polygon
   - Columns 13-(13+2*maxPolygonVertices): Vertex coordinates (x0,y0,x1,y1,...), NaN padded
   - Last 4 columns: Quadrature weights
   Total: 16 + 1 + 2*maxPolygonVertices columns
"""

import os, time, pickle, logging, concurrent.futures, gc
from fcm_quadrature.data_generation.parameters import ProjectParameter
import numpy as np

# Import moment fitting

from fcm_quadrature.data_generation.moment_fitting import MomentFittingQuadrature, fit_quadrature_weights


class Dataset():
    def __init__(self, name, projectName, dataFolderName, appendDataset):
        self.name = name
        self.folderName = dataFolderName + '/' + projectName
        self.path = self.getPath()
        self.csvPath = self.path + '.csv'
        os.makedirs(self.folderName, exist_ok=True)
        if not appendDataset:
            if os.path.exists(self.csvPath):
                os.remove(self.csvPath)
        self.clear()

    def getPath(self):
        return self.folderName + '/' + self.name

    def clear(self):
        self.inputData = []
        self.outputData = []

    def append(self, input, output):
        self.inputData.append(input)
        self.outputData.append(output)

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        with open(self.path, 'rb') as f:
            tempdict = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(tempdict)

    def saveCSV(self):
        if self.outputData:
            inputDim = self.inputData[0].size
            if isinstance(self.outputData[0], float):
                outputDim = 1
            else:
                outputDim = self.outputData[0].size
            tempArray = np.empty((len(self.inputData), inputDim + outputDim))
            with open(self.csvPath, 'a') as csvFile:
                for idxRow, row in enumerate(self.inputData):
                    tempArray[idxRow, 0:inputDim] = row.flatten()
                    if outputDim == 1:
                        tempArray[idxRow, inputDim] = self.outputData[idxRow]
                    else:
                        tempArray[idxRow, inputDim: inputDim + outputDim] = self.outputData[idxRow]
                np.savetxt(csvFile, tempArray, delimiter=',', fmt='%.16f')

    def saveLabel(self, label: str):
        labelPath = self.path + '_label.txt'
        with open(labelPath, 'w') as labelFile:
            print(label, file=labelFile)


class idxIterator():
    def __init__(self, maxValue, superIterator=False, startValue=0):
        self.maxValue = maxValue
        self.superIterator = superIterator
        self.startValue = startValue
        self.value = startValue

    def step(self):
        if self.value + 1 < self.maxValue:
            self.value += 1
            finished = False
        else:
            self.value = self.startValue
            if self.superIterator != False:
                finished = self.superIterator.step()
            else:
                finished = True
        return finished

    def reset(self):
        self.value = self.startValue


class Project():
    def __init__(self, par: ProjectParameter):
        self.par = par
        self.path = par.projectName + '.' + 'Project'

        if os.path.exists(self.path):
            self.load()
            self.logger.info('checkpoint reloaded.')
            self.logProgress(None)
        else:
            logging.basicConfig(
                filename=self.par.logName,
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
            )
            self.logger = logging.getLogger()
            self.dataset = Dataset(
                name=self.par.datasetName,
                projectName=self.par.projectName,
                dataFolderName=self.par.dataFolderName,
                appendDataset=self.par.appendDataset,
            )
            
            if os.path.exists(par.tempFolderName):
                import shutil
                shutil.rmtree(par.tempFolderName)
            os.makedirs(self.par.tempFolderName, exist_ok=True)
            
            from fcm_quadrature.data_generation.mesh import Point, PointList, Line, LineList, Rectangle
            
            self.boundary = Rectangle(
                par.meshSideLength*par.scalingFactorX,
                par.meshSideLength*par.scalingFactorY,
            )
            self.cell = Rectangle(
                par.cellSideLength*par.scalingFactorX,
                par.cellSideLength*par.scalingFactorY,
            )
            
            self.getTargets()
            self.dataset.saveLabel(self.targetLabel)

            l = self.par.cellSideLength/2

            if self.par.useAllEdgesAsStart:
                # Generate all edge pair configurations
                # Each edge can be a start edge, with the other 3 as end edges
                self.allCutConfigs = []  # List of (startPoint, endPoint) tuples
                self.allCutConfigIDs = []  # List of job ID strings

                for startEdgeIdx in range(4):
                    # Sample points on start edge
                    startEdge = LineList(self.cell.allEdges[startEdgeIdx])
                    startPoints = startEdge.samplePoints(
                        self.par.numSamplesStartEdge,
                        method=self.par.edgePointMethod,
                        startRatio=self.par.minEdgeLengthRatio,
                    )

                    # End edges are the other 3 edges
                    endEdgeIndices = [i for i in range(4) if i != startEdgeIdx]
                    endEdges = LineList(*[self.cell.allEdges[i] for i in endEdgeIndices])
                    endPoints = endEdges.samplePoints(
                        self.par.numSamplesEndEdge,
                        method=self.par.edgePointMethod,
                        startRatio=self.par.minEdgeLengthRatio,
                    )

                    # Generate all combinations for this start edge
                    for si, sp in enumerate(startPoints):
                        for ei, ep in enumerate(endPoints):
                            self.allCutConfigs.append((sp, ep))
                            self.allCutConfigIDs.append(f'e{startEdgeIdx}_{si}_{ei}')

                self.idxCutConfig = idxIterator(len(self.allCutConfigs))
                self.logger.info(f'All-edges mode: {len(self.allCutConfigs)} total configurations')

            else:
                # Original mode: only edge 0 as start edge
                allStartEdge = LineList(self.cell.allEdges[0])
                allEndEdge = LineList(*self.cell.allEdges[1:])

                self.allStartPoint = allStartEdge.samplePoints(
                    self.par.numSamplesStartEdge,
                    method=self.par.edgePointMethod,
                    startRatio=self.par.minEdgeLengthRatio,
                )
                self.allEndPoint = allEndEdge.samplePoints(
                    self.par.numSamplesEndEdge,
                    method=self.par.edgePointMethod,
                    startRatio=self.par.minEdgeLengthRatio,
                )
                self.idxStartPoint = idxIterator(self.par.numSamplesStartEdge)
                self.idxEndPoint = idxIterator(
                    len(self.allEndPoint),
                    superIterator=self.idxStartPoint
                )

            if self.par.cutSectionType == 'arc':
                self.allDirection = np.array([-1, 1])
                self.allRadiusRatio = np.linspace(
                    self.par.minRadiusRatio,
                    self.par.maxRadiusRatio,
                    num=self.par.numRadiusRatio
                )
                if self.par.useAllEdgesAsStart:
                    self.idxDirection = idxIterator(
                        2,
                        superIterator=self.idxCutConfig
                    )
                else:
                    self.idxDirection = idxIterator(
                        2,
                        superIterator=self.idxEndPoint
                    )
                self.idxRadiusRatio = idxIterator(
                    self.par.numRadiusRatio,
                    superIterator=self.idxDirection
                )

            self.jobCounter = 0
            self.errorCounter = 0

            self.save()

    def getTargets(self):
        from fcm_quadrature.data_generation.mesh import Point, PointList
        def addItem(newTarget, itemName):
            startIdx = len(self.targets)
            self.targets += newTarget
            endIdx = len(self.targets)
            idxItem = len(self.targetLabel)
            self.targetLabel += f'np.arange({startIdx}, {endIdx}), # {idxItem}, {itemName} \n'

        self.targets = PointList()
        self.targetLabel = str()

        if 'cell vertices' in self.par.targets:
            newTarget = self.cell.allVertices.copy()
            addItem(newTarget, 'cell vertices')

        if 'center point' in self.par.targets:
            newTarget = PointList(Point([0, 0]))
            addItem(newTarget, 'center point')

        if 'edge Mid Points' in self.par.targets:
            newTarget = PointList()
            for edge in self.cell.allEdges:
                newTarget.append(edge.getMidPoint())
            addItem(newTarget, 'edge Mid Points')

        if 'Gauss Points' in self.par.targets:
            xi = 1/np.sqrt(3)
            newTarget = PointList(
                    Point([xi, xi]),
                    Point([-xi, xi]),
                    Point([-xi, -xi]),
                    Point([xi, -xi]),
                )
            addItem(newTarget, 'Gauss Points')

        if 'interior Mid Points' in self.par.targets:
            l = self.par.cellSideLength/4
            newTarget = PointList(
                    Point([l, l]),
                    Point([-l, l]),
                    Point([-l, -l]),
                    Point([l, -l]),
                )
            addItem(newTarget, 'interior Mid Points')

        if 'evenly spaced points on edge' in self.par.targets:
            for numTarget in self.par.numEvenlySpacedTarget:
                newTarget = self.cell.allEdges.samplePoints(
                        numTarget,
                        method='linear',
                        startRatio=1/(numTarget + 1),
                    )
                addItem(newTarget, f'evenly spaced {numTarget} on each edge')

        if '2 points at given ratio' in self.par.targets:
            for targetRatio in self.par.ratio_2TargetPoints:
                newTarget = self.cell.allEdges.samplePoints(
                        2,
                        method='linear',
                        startRatio=targetRatio,
                    )
                addItem(newTarget, f'2 points at {targetRatio:.3e}')

    def generateSubList(self, finished=False):
        subList = []
        jobIDList = []
        while len(subList) < self.par.subListLength and not finished:
            if self.par.cutSectionType == 'line':
                if self.par.useAllEdgesAsStart:
                    # All-edges mode: use pre-computed configurations
                    startPoint, endPoint = self.allCutConfigs[self.idxCutConfig.value]
                    subList.append((startPoint, endPoint))
                    jobIDList.append(self.allCutConfigIDs[self.idxCutConfig.value])
                    finished = self.idxCutConfig.step()
                else:
                    # Original mode
                    subList.append(
                        (
                            self.allStartPoint[self.idxStartPoint.value],
                            self.allEndPoint[self.idxEndPoint.value],
                        )
                    )
                    jobIDList.append(
                        f'{self.idxStartPoint.value}_{self.idxEndPoint.value}')
                    finished = self.idxEndPoint.step()
            elif self.par.cutSectionType == 'arc':
                if self.par.useAllEdgesAsStart:
                    startPoint, endPoint = self.allCutConfigs[self.idxCutConfig.value]
                    subList.append(
                        (
                            startPoint,
                            endPoint,
                            self.allDirection[self.idxDirection.value],
                            self.allRadiusRatio[self.idxRadiusRatio.value],
                        )
                    )
                    jobIDList.append(f'{self.allCutConfigIDs[self.idxCutConfig.value]}_'
                        f'{self.idxDirection.value}_{self.idxRadiusRatio.value}')
                else:
                    subList.append(
                        (
                            self.allStartPoint[self.idxStartPoint.value],
                            self.allEndPoint[self.idxEndPoint.value],
                            self.allDirection[self.idxDirection.value],
                            self.allRadiusRatio[self.idxRadiusRatio.value],
                        )
                    )
                    jobIDList.append(f'{self.idxStartPoint.value}_{self.idxEndPoint.value}_'
                        f'{self.idxDirection.value}_{self.idxRadiusRatio.value}')
                finished = self.idxRadiusRatio.step()
        self.logger.info(
            f'subList generated: {jobIDList}'
        )
        return finished, subList, jobIDList

    def parallelExecute(self):
        if self.par.useAllEdgesAsStart:
            self.logger.info(
                f'start execution (all-edges mode), total configs: {len(self.allCutConfigs)}'
            )
        else:
            self.logger.info(
                f'start execution, idxStartPoint range: [{self.idxStartPoint.value}, {self.idxStartPoint.maxValue})'
            )

        # Calculate total number of jobs
        if self.par.useAllEdgesAsStart:
            if self.par.cutSectionType == 'line':
                total_jobs = len(self.allCutConfigs)
            else:  # arc
                total_jobs = (len(self.allCutConfigs) *
                             len(self.allDirection) * self.par.numRadiusRatio)
        else:
            if self.par.cutSectionType == 'line':
                total_jobs = self.par.numSamplesStartEdge * len(self.allEndPoint)
            else:  # arc
                total_jobs = (self.par.numSamplesStartEdge * len(self.allEndPoint) *
                             len(self.allDirection) * self.par.numRadiusRatio)
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            self.logger.warning('tqdm not available, progress bar disabled')
        
        # Create progress bar
        if use_tqdm:
            pbar = tqdm(total=total_jobs, desc="Generating dataset", 
                       unit="jobs", ncols=100, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.par.numWorkers) as executor:
            finished = False
            while not finished:
                startTimeSubList = time.perf_counter()
                finished, subList, jobIDList = self.generateSubList()
                
                allFutures = []
                for jobID, cutParameters in zip(jobIDList, subList):
                    job = Job(self, jobID, cutParameters)
                    future = executor.submit(job.execute)
                    allFutures.append(future)
                
                for future in concurrent.futures.as_completed(allFutures):
                    error, result = future.result()
                    if error:
                        logInfo = result
                        self.errorCounter += 1
                    else:
                        input, output, logInfo = result
                        self.dataset.append(input, output)
                        self.jobCounter += 1
                    self.logExecutionError(logInfo)
                    
                    # Update progress bar
                    if use_tqdm:
                        pbar.update(1)
                        pbar.set_postfix({
                            'success': self.jobCounter,
                            'failed': self.errorCounter,
                            'rate': f'{self.jobCounter/(self.jobCounter+self.errorCounter)*100:.1f}%' if (self.jobCounter+self.errorCounter) > 0 else '0%'
                        })
                    
                self.checkPoint()
                finishTimeSubList = time.perf_counter()
                speed = len(subList)/(finishTimeSubList - startTimeSubList)
                self.logProgress(speed)
        
        if use_tqdm:
            pbar.close()
            
        executor.shutdown()
        self.cleanUp()

    def checkPoint(self):
        self.dataset.saveCSV()
        self.dataset.clear()
        self.save()

    def cleanUp(self):
        self.dataset.saveCSV()
        self.dataset.clear()
        self.logger.info(
            f'all {self.jobCounter} jobs are saved to {self.dataset.csvPath}')
        os.remove(self.path)

    def logProgress(self, speed):
        self.logger.info(f'{self.jobCounter} jobs finished, {self.errorCounter} jobs failed.')
        if speed is not None:
            self.logger.info(f'speed of current sublist: {speed:.2f}jobs/s.')
        self.logger.info(f'progress:')
        if self.par.useAllEdgesAsStart:
            self.logger.info(
                f'# cutConfig idx = {self.idxCutConfig.value}/{len(self.allCutConfigs)}'
            )
        else:
            self.logger.info(
                f'# startPoint idx = {self.idxStartPoint.value}, '
                f'endPoint idx = {self.idxEndPoint.value}'
            )
        if self.par.cutSectionType == 'arc':
            self.logger.info(
                f'# direction idx = {self.idxDirection.value}, '
                f'radius ratio idx = {self.idxRadiusRatio.value}'
            )

    def logExecutionError(self, logInfo):
        for item in logInfo:
            self.logger.warning(item)

    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        self.logger.info(f'check point saved, job count sofar: {self.jobCounter}.')

    def load(self):
        with open(self.path, 'rb') as f:
            tempdict = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(tempdict)


class Job():
    def __init__(self, project: Project, jobID, cutParameters):
        self.par = project.par
        self.targets = project.targets
        self.boundary = project.boundary
        self.cell = project.cell
        self.jobID = jobID
        self.cutParameters = cutParameters

    def execute(self):
        '''
        Execute job: generate cut cell and compute quadrature weights.

        Output:
            error: bool - True if execution failed
            result: if error, return logInfo
                    if success, return (inputData, outputData, logInfo)
                    where:
                        inputData: np.ndarray - [12 signed distances, area fraction]
                        outputData: np.ndarray - [4 quadrature weights]
                        logInfo: list - warning/error messages
        '''
        logInfo = []

        # Initialize mesh object
        from fcm_quadrature.data_generation.mesh import Mesh
        self.mesh = Mesh(self.par, self.boundary, self.cell, self.targets)
        self.mesh.setCutParameters(*self.cutParameters)

        if self.par.generateInputOnly:
            # Generate input only mode
            weights = np.zeros(4)
            error = False
        else:
            # Generate mid points
            self.mesh.generateMidPoints()
            
            # Check if all middle points are within the cell (for arc cuts)
            if self.par.cutSectionType == 'arc':
                error = self.mesh.verifyMidPoints()
                if error:
                    logInfo.append(
                        f'job: {self.jobID}, middle point(s) out of the cell')
                    return error, logInfo
            
            # Generate cut configuration
            self.mesh.generateCutConfig()
            
            # ===== COMPUTE QUADRATURE WEIGHTS USING MOMENT FITTING =====
            try:
                # Get cut cell vertices as numpy array
                vertices = self.mesh.getCutVertices(self.cell)
                vertices_array = np.array([[v.getX(), v.getY()] for v in vertices])

                # Initialize moment fitting
                mfq = MomentFittingQuadrature()

                # Get standard 2x2 Gauss quadrature points in [-1,1]^2
                # For finite cell method, these points stay at their standard positions
                # The cell is [-1,1]^2 (cellSideLength=2, centered at origin)
                quad_points, _ = mfq.get_standard_gauss_points_2d(order=2)

                # NOTE: quad_points are at standard positions: ±1/√3 ≈ ±0.577
                # These are the correct positions for 2x2 Gauss quadrature on [-1,1]^2
                # The moment fitting computes weights that integrate correctly over
                # the cut cell polygon, keeping the quadrature points fixed.

                # Compute weights using moment fitting
                weights, verification = fit_quadrature_weights(
                    quad_points,
                    vertices_array,
                    verify=True
                )

                # Check if weights are acceptable
                tolerance = self.par.weightVerificationTolerance
                if not verification['is_valid'] or verification['max_error'] > tolerance:
                    error = True
                    logInfo.append(
                        f'job: {self.jobID}, moment fitting verification failed')
                    logInfo.append(
                        f'  max moment error: {verification["max_error"]:.3e}')
                    return error, logInfo

                error = False

            except Exception as e:
                error = True
                logInfo.append(f'job: {self.jobID}, moment fitting exception: {str(e)}')
                return error, logInfo

        # Prepare input data: [12 signed distances]
        # Note: area fraction removed as per user request
        signedDistances = self.mesh.getSignedCutDistances()

        if self.par.savePolygonVertices:
            # Include polygon vertices in input data
            # Flatten vertices and pad with NaN to fixed size
            vertices = self.mesh.getCutVertices(self.cell)
            vertices_flat = []
            for v in vertices:
                vertices_flat.extend([v.getX(), v.getY()])
            # Pad with NaN to maxPolygonVertices * 2 values
            max_coords = self.par.maxPolygonVertices * 2
            while len(vertices_flat) < max_coords:
                vertices_flat.append(np.nan)
            vertices_array = np.array(vertices_flat[:max_coords])
            # Also store the actual number of vertices
            num_vertices = np.array([len(vertices)])
            inputData = np.concatenate((signedDistances, num_vertices, vertices_array))
        else:
            inputData = signedDistances

        # Output data: 4 quadrature weights
        outputData = weights

        result = (inputData, outputData, logInfo)
        return error, result