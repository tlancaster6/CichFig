from helpers.legacy.LogParser import LogParser as LP
import numpy as np
from skimage import morphology
from types import SimpleNamespace
import datetime
import sys

np.warnings.filterwarnings('ignore')

class DepthAnalyzer():
	def __init__(self, projFileManager):
		self.projFileManager = projFileManager
		self.lp = LP(self.projFileManager.localLogfile)
		self._loadData()
		self.goodPixels = (self.tray_r[2] - self.tray_r[0])*(self.tray_r[3] - self.tray_r[1])

	def _loadData(self):
		# Loads depth tray information and smoothedDepthData from files that have already been downloaded

		# If tray attribute already exists, exit
		try:
			self.tray_r
		except AttributeError:
			with open(self.projFileManager.localTrayFile) as f:
				line = next(f)
				tray = line.rstrip().split(',')
				self.tray_r = [int(x) for x in tray]
			
		try:
			self.smoothDepthData
		except AttributeError:		
			self.smoothDepthData = np.load(self.projFileManager.localSmoothDepthFile)
			self.smoothDepthData[:,:self.tray_r[0],:] = np.nan
			self.smoothDepthData[:,self.tray_r[2]:,:] = np.nan
			self.smoothDepthData[:,:,:self.tray_r[1]] = np.nan
			self.smoothDepthData[:,:,self.tray_r[3]:] = np.nan

	def returnBowerLocations(self, t0, t1, cropped=False):
		# Returns 2D numpy array using thresholding and minimum size data to identify bowers
		# Pits = -1, Castle = 1, No bower = 0
		# threshold and min pixels will be automatically determined if not supplied
		# 

		# Check times are good
		self._checkTimes(t0,t1)

		# Identify total height change and time change
		totalHeightChange = self.returnHeightChange(t0, t1, masked=False, cropped=False)
		timeChange = t1 - t0

		# Determine threshold and minimum size of bower to use based upon timeChange
		if timeChange.total_seconds() < 7300:  # 2 hours or less
			totalThreshold = self.projFileManager.hourlyDepthThreshold
			minPixels = self.projFileManager.hourlyMinPixels
		elif timeChange.total_seconds() < 129600:  # 2 hours to 1.5 days
			totalThreshold = self.projFileManager.dailyDepthThreshold
			minPixels = self.projFileManager.dailyMinPixels
		else:  # 1.5 days or more
			totalThreshold = self.projFileManager.totalDepthThreshold
			minPixels = self.projFileManager.totalMinPixels

		tCastle = np.where(totalHeightChange >= totalThreshold, True, False)
		tCastle = morphology.remove_small_objects(tCastle, minPixels).astype(int)

		tPit = np.where(totalHeightChange <= -1 * totalThreshold, True, False)
		tPit = morphology.remove_small_objects(tPit, minPixels).astype(int)

		bowers = tCastle - tPit
		if cropped:
			bowers = bowers[self.tray_r[0]:self.tray_r[2], self.tray_r[1]:self.tray_r[3]]

		return bowers
		
	def returnHeight(self, t, cropped=False):

		# Check times are good
		self._checkTimes(t)

		# Load necessary data

	   	# Find closest frames to desired times
		try:
			first_index = max([False if x.time<=t else True for x in self.lp.frames].index(True) - 1, 0) #This ensures that we get overnight changes when kinect wasn't running
		except ValueError:
			if t > self.lp.frames[-1].time:
				first_index = -1
			else:
				first_index = 0

		change = self.smoothDepthData[first_index]
		
		if cropped:
			change = change[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]]
			
		return change	

	def returnHeightChange(self, t0, t1, masked=False, cropped=False):
		# Check times are good
		self._checkTimes(t0,t1)
		
		# Find closest frames to desired times
		try:
			first_index = max([False if x.time<=t0 else True for x in self.lp.frames].index(True) - 1, 0) #This ensures that we get overnight changes when kinect wasn't running
		except ValueError:
			if t0 > self.lp.frames[-1].time:
				first_index = -1
			else:
				first_index = 0

		try:
			last_index = max([False if x.time<=t1 else True for x in self.lp.frames].index(True) - 1, 0)
		except ValueError:
			last_index = len(self.lp.frames) - 1
			
		change = self.smoothDepthData[first_index] - self.smoothDepthData[last_index]
		
		if masked:
			change[self.returnBowerLocations(t0, t1) == 0] = 0

		if cropped:
			change = change[self.tray_r[0]:self.tray_r[2],self.tray_r[1]:self.tray_r[3]]
			
		return change

	def returnVolumeSummary(self, bowerLocations, heightChange):

		pixelLength = self.projFileManager.pixelLength
		bowerIndex_pixels = int(self.goodPixels*self.projFileManager.bowerIndexFraction)

		heightChangeAbs = heightChange.copy()
		heightChangeAbs = np.abs(heightChangeAbs)

		outData = SimpleNamespace()
		# Get data
		outData.projectID = self.lp.projectID
		outData.absoluteVolume = np.nansum(heightChangeAbs)*pixelLength**2
		outData.summedVolume = np.nansum(heightChange)*pixelLength**2
		outData.castleArea = np.count_nonzero(bowerLocations == 1)*pixelLength**2
		outData.pitArea = np.count_nonzero(bowerLocations == -1)*pixelLength**2
		outData.castleVolume = np.nansum(heightChange[bowerLocations == 1])*pixelLength**2
		outData.pitVolume = np.nansum(heightChange[bowerLocations == -1])*-1*pixelLength**2
		outData.bowerVolume = outData.castleVolume + outData.pitVolume

		flattenedData = heightChangeAbs.flatten()
		sortedData = np.sort(flattenedData[~np.isnan(flattenedData)])
		threshold = sortedData[-1*bowerIndex_pixels]
		thresholdCastleVolume = np.nansum(heightChangeAbs[(bowerLocations == 1) & (heightChangeAbs > threshold)])
		thresholdPitVolume = np.nansum(heightChangeAbs[(bowerLocations == -1) & (heightChangeAbs > threshold)])

		outData.bowerIndex = (thresholdCastleVolume - thresholdPitVolume)/(thresholdCastleVolume + thresholdPitVolume)
		
		return outData

	def _checkTimes(self, t0, t1 = None):
		if t1 is None:
			if type(t0) != datetime.datetime:
				raise Exception('Timepoints to must be datetime.datetime objects')
			return
		# Make sure times are appropriate datetime objects
		if type(t0) != datetime.datetime or type(t1) != datetime.datetime:
			raise Exception('Timepoints to must be datetime.datetime objects')
		if t0 > t1:
			print('Warning: Second timepoint ' + str(t1) + ' is earlier than first timepoint ' + str(t0), file = sys.stderr)
