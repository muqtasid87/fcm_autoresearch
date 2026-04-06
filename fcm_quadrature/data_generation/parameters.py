"""
Updated parameter.py for moment fitting data generation.

New parameters added for moment fitting:
- weightVerificationTolerance: Tolerance for verifying computed weights
- generateDebugVisualizations: Whether to generate debug plots
- debugVisualizationFolder: Where to save debug visualizations
"""

import numpy as np

class ProjectParameter():
    def __init__(self, projectName, datasetName):

        # ===== Cut cell sampling parameters =====
        # these 2 parameters must be odd numbers
        self.numSamplesStartEdge = 15
        self.numSamplesEndEdge = 15
        
        self.numRadiusRatio = 10  # needed for arc cut

        # ===== Project naming and organization =====
        self.projectName = projectName
        self.datasetName = datasetName
        self.tempFolderName = 'temp'
        self.dataFolderName = 'Data'
        self.meshFileFormat = 'inp'
        self.logName = 'log.txt'

        # ===== Debug and development options =====
        self.keepDebugData = False
        self.generateInputOnly = False  # If True, skip weight computation

        # ===== Mesh parameters =====
        self.meshSize = 5e-2
        self.cellSideLength = 2
        self.meshSideLength = 3  # larger square for generating the mesh
        self.minEdgeLengthRatio = 2e-3
        self.edgePointMethod = 'log'

        # ===== Scaling parameters =====
        self.scalingFactorX = 1
        self.scalingFactorY = 1
        self.appendDataset = True  # do not delete existing dataset with same name

        # ===== Target points for distance computation =====
        self.targets = [
            'cell vertices',
            # 'center point',
            # 'edge Mid Points',
            # 'Gauss Points',
            # 'interior Mid Points',
            # 'evenly spaced points on edge',
            '2 points at given ratio',
        ]
        self.numEvenlySpacedTarget = [2, 3, 4, 5]
        self.ratio_2TargetPoints = [1e-3]

        # ===== Cut section configuration =====
        self.midPointRatios = np.array([])  # np.array([0.5])
        self.cutSectionType = 'line'  # 'line' or 'arc'

        # If True, start points can come from all 4 edges (not just edge 0)
        # This generates more diverse cut configurations
        self.useAllEdgesAsStart = False

        # ===== Output configuration =====
        # If True, save polygon vertices (cut cell geometry) to CSV
        # Format: [signed_distances, (optional: vertices), weights]
        self.savePolygonVertices = False
        # Maximum number of vertices to save (polygon padded with NaN if fewer)
        self.maxPolygonVertices = 6  # Most cut cells have 3-6 vertices

        # ===== Arc cut parameters (if cutSectionType == 'arc') =====
        self.minRadiusRatio = 0.5
        self.maxRadiusRatio = 10

        # ===== NEW: Moment fitting parameters =====
        # Tolerance for verifying that computed weights satisfy moment equations
        self.weightVerificationTolerance = 1e-10
        
        # Whether to generate debug visualizations (slow, use for debugging only)
        self.generateDebugVisualizations = False
        
        # Folder for debug visualizations
        self.debugVisualizationFolder = 'debug_visualizations'

        # ===== Parallel execution parameters =====
        self.numWorkers = 6
        self.subListLength = self.numWorkers * 5

        # ===== REMOVED: Eigenvalue-specific parameters =====
        # The following parameters are no longer needed for moment fitting:
        # - numAI (adaptive integration for eigenvalue)
        # - numAIEigValThreshold
        # - maxAI
        # - thresholdRepetition
        # - maxRepetition
        # - toleranceRepetition


# configurations for different use cases
class QuickTestConfig(ProjectParameter):
    """Configuration for quick testing (small dataset)."""
    def __init__(self, projectName='QuickTest', datasetName='Test'):
        super().__init__(projectName, datasetName)
        self.numSamplesStartEdge = 5
        self.numSamplesEndEdge = 5
        self.numWorkers = 2
        self.subListLength = 4
        self.generateDebugVisualizations = True  # Enable for testing


class ProductionConfig(ProjectParameter):
    """Configuration for production runs (large dataset)."""
    def __init__(self, projectName='Mesh2D', datasetName='Training'):
        super().__init__(projectName, datasetName)
        self.numSamplesStartEdge = 399  
        self.numSamplesEndEdge = 399
        self.numWorkers = 12
        self.subListLength = self.numWorkers * 5
        self.generateDebugVisualizations = False  # Disable for speed


class DebugConfig(ProjectParameter):
    """Configuration for debugging with visualizations."""
    def __init__(self, projectName='Debug', datasetName='Debug'):
        super().__init__(projectName, datasetName)
        self.numSamplesStartEdge = 3
        self.numSamplesEndEdge = 3
        self.numWorkers = 1
        self.subListLength = 1
        self.generateDebugVisualizations = True
        self.keepDebugData = True


class MillionSamplesConfig(ProjectParameter):
    """Configuration for generating ~1 million samples.

    With all-edges mode: 4 × 289 × (3 × 289) = 1,001,412 samples
    """
    def __init__(self, projectName='MomentFit_1M_NoVertices', datasetName='Training_1M_NoVertices'):
        super().__init__(projectName, datasetName)
        self.numSamplesStartEdge = 289
        self.numSamplesEndEdge = 289
        self.useAllEdgesAsStart = True
        self.savePolygonVertices = True
        self.numWorkers = 128
        self.subListLength = self.numWorkers * 10
        self.weightVerificationTolerance = 1e-16
        self.generateDebugVisualizations = True
