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

        # ===== Arc input features =====
        # If True, append (radius_ratio, direction) to distance columns in CSV
        self.includeArcFeatures = False

    def configure_targets(self, num_distances, strategy='auto'):
        """Configure target points to produce the desired number of input distances.

        '2 points at given ratio' produces 8 points per ratio value (2 per edge × 4 edges).
        Formula (log strategy):  num_distances = 4 (vertices) + 8 * k
        Formula (even strategy): num_distances = 4 (vertices) + 4 * n_per_edge

        For counts that satisfy both formulas (12, 20, 28, ...) the default strategy='auto'
        picks 'log'. Use strategy='even' to force evenly-spaced placement instead.

        Supported counts:
          log:  12, 20, 28, ... (4 + 8k, k>=1)
          even: 8, 12, 16, 20, 24, 28, ... (4 + 4n, n>=1)
          4: vertices only (no strategy needed)

        Parameters
        ----------
        num_distances : int
            Desired number of input distance features.
        strategy : str, optional
            Placement strategy: 'log' (near-boundary log-spaced ratios),
            'even' (uniform spacing), or 'auto' (log if possible, else even).

        Returns
        -------
        self
            For chaining.
        """
        if num_distances == 4:
            self.targets = ['cell vertices']
            self.ratio_2TargetPoints = []
        elif strategy in ('log', 'auto') and (num_distances - 4) % 8 == 0 and num_distances > 4:
            k = (num_distances - 4) // 8
            self.targets = ['cell vertices', '2 points at given ratio']
            if k == 1:
                self.ratio_2TargetPoints = [1e-3]
            else:
                self.ratio_2TargetPoints = np.logspace(-3, np.log10(0.4), k).tolist()
        elif strategy in ('even', 'auto') and (num_distances - 4) % 4 == 0 and num_distances > 4:
            n_per_edge = (num_distances - 4) // 4
            self.targets = ['cell vertices', 'evenly spaced points on edge']
            self.numEvenlySpacedTarget = [n_per_edge]
            self.ratio_2TargetPoints = []
        else:
            raise ValueError(
                f"num_distances={num_distances} with strategy='{strategy}' not supported. "
                f"Use 4+4k (even) or 4+8k (log) for integer k >= 1, or 4 for vertices only."
            )
        return self

    def get_num_distances(self):
        """Compute the number of distance features from current target config."""
        count = 0
        if 'cell vertices' in self.targets:
            count += 4
        if 'center point' in self.targets:
            count += 1
        if 'edge Mid Points' in self.targets:
            count += 4
        if 'Gauss Points' in self.targets:
            count += 4
        if 'interior Mid Points' in self.targets:
            count += 4
        if 'evenly spaced points on edge' in self.targets:
            count += sum(n * 4 for n in self.numEvenlySpacedTarget)
        if '2 points at given ratio' in self.targets:
            count += 8 * len(self.ratio_2TargetPoints)
        return count


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


class ArcConfig(ProjectParameter):
    """Configuration for arc-cut data generation."""
    def __init__(self, projectName='ArcCut', datasetName='Arc_Training',
                 num_samples=15, num_radius=10):
        super().__init__(projectName, datasetName)
        self.cutSectionType = 'arc'
        self.numSamplesStartEdge = num_samples
        self.numSamplesEndEdge = num_samples
        self.numRadiusRatio = num_radius
        self.minRadiusRatio = 0.5
        self.maxRadiusRatio = 10
        self.useAllEdgesAsStart = True


class ArcWithFeaturesConfig(ArcConfig):
    """Arc config that also outputs curvature features (radius, direction)."""
    def __init__(self, projectName='ArcCut_Features', datasetName='Arc_Features_Training',
                 num_samples=15, num_radius=10):
        super().__init__(projectName, datasetName, num_samples, num_radius)
        self.includeArcFeatures = True


class CombinedConfig(ProjectParameter):
    """Configuration for generating both line + arc data.

    Runs line generation first, then arc generation, concatenating outputs.
    Use with generate_data.py --cut-type both.
    """
    def __init__(self, projectName='Combined', datasetName='Combined_Training',
                 num_samples=15, num_radius=10):
        super().__init__(projectName, datasetName)
        self.cutSectionType = 'both'
        self.numSamplesStartEdge = num_samples
        self.numSamplesEndEdge = num_samples
        self.numRadiusRatio = num_radius
        self.useAllEdgesAsStart = True
