# -*- coding: utf-8 -*-

#--------------------------------------------------------------------#
#                                                                    #
# Copyright (C) 2023 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
#                                                                    #
#--------------------------------------------------------------------#


import ctypes
import os
import sys
import time
import weakref
from platform import system

## \cond INTERNALDOC
## Stores if the current Python version is 3 or higher
isPython3 = sys.version_info[0] == 3
## \endcond

## Stores if NumPy could be found.
# \ingroup SLMDisplayPython
supportNumPy = True

try:
    import numpy
except:
    supportNumPy = False

if isPython3:
    sys.path.append(os.path.dirname(__file__))
    from heds_types import LoadFlags, ShowFlags, ErrorCode, State, ZernikeValues, ApplyDataHandleValue, SLMPreviewFlags, Format, WavefrontcompensationFlags
    del sys.path[-1]
else:
    from .heds_types import LoadFlags, ShowFlags, ErrorCode, State, ZernikeValues, ApplyDataHandleValue, SLMPreviewFlags, Format, WavefrontcompensationFlags

## Creates a field for a given type description for either numpy or ctypes.
# \ingroup SLMDisplayPython
# \param width The width of the generated array. If \p width or \p height is zero, the slm size will be used.
# \param height The height of the generated array. If \p width or \p height is zero, the slm size will be used.
# \param elementSizeInBytes The size of each element in bytes, for example 4 for an integer.
# \param elementIsReal Defines if the created array will be one of integers or real numbers.
# \param useNumPy When true, a numpy array will be created when it is supported. Otherwise a ctypes array will be created.
# \return A 2d array of the given type based on numpy or ctypes.
def createFieldByType(width, height, elementSizeInBytes, elementIsReal, useNumPy):

    width = int(width)
    height = int(height)

    assert width > 0 and height > 0
    assert elementSizeInBytes > 0

    if useNumPy:
        assert supportNumPy, "NumPy could not be found"

        elementType = None

        if elementIsReal:
            if elementSizeInBytes == 4:
                elementType = numpy.single
            elif elementSizeInBytes == 8:
                elementType = numpy.double
        else:
            if elementSizeInBytes == 1:
                elementType = numpy.ubyte
            elif elementSizeInBytes == 4:
                elementType = numpy.uintc

        assert elementType is not None, "The given format - size:" + str(elementSizeInBytes) + "  real:" + str(elementIsReal) + " - is not supported."

        return numpy.empty((height, width), elementType)

    else:
        elementType = None

        if elementIsReal:
            if elementSizeInBytes == 4:
                elementType = ctypes.c_float
            elif elementSizeInBytes == 8:
                elementType = ctypes.c_double
        else:
            if elementSizeInBytes == 1:
                elementType = ctypes.c_ubyte
            elif elementSizeInBytes == 4:
                elementType = ctypes.c_uint

        assert elementType is not None, "The given format - size:" + str(elementSizeInBytes) + "  real:" + str(elementIsReal) + " - is not supported."

        return ((elementType * width) * height)()

## Creates an unsigned byte array for either numpy or ctypes
# \ingroup SLMDisplayPython
# \param width The width of the generated array.
# \param height The height of the generated array.
# \param useNumPy When true, a numpy array will be created when it is supported. Otherwise a ctypes array will be created.
# \return A 2d array of unsigned bytes based on numpy or ctypes.
def createFieldUChar(width, height, useNumPy = True):
    assert width > 0 and height > 0, "Invalid field size"

    return createFieldByType(width, height, 1, False, useNumPy and supportNumPy)

## Creates an array of single real values for either numpy or ctypes.
# \ingroup SLMDisplayPython
# \param width The width of the generated array.
# \param height The height of the generated array.
# \param useNumPy When true, a numpy array will be created when it is supported. Otherwise a ctypes array will be created.
# \return A 2d array of floats based on numpy or ctypes.
def createFieldSingle(width, height, useNumPy = True):
    assert width > 0 and height > 0, "Invalid field size"

    return createFieldByType(width, height, 4, True, useNumPy and supportNumPy)

## Creates an array of double real values for either numpy or ctypes.
# \ingroup SLMDisplayPython
# \param width The width of the generated array.
# \param height The height of the generated array.
# \param useNumPy When true, a numpy array will be created when it is supported. Otherwise a ctypes array will be created.
# \return A 2d array of doubles based on numpy or ctypes.
def createFieldDouble(width, height, useNumPy = True):
    assert width > 0 and height > 0, "Invalid field size"

    return createFieldByType(width, height, 8, True, useNumPy and supportNumPy)

## Returns the width of the given field.
# \ingroup SLMDisplayPython
# \param field The field whose width we want.
# \return Zero when there was an error. Otherwise a number greater than zero.
def width(field):
    if supportNumPy and isinstance(field, numpy.ndarray):
        assert field.size > 0, "The given array has no size"

    else:
        assert isinstance(field, ctypes.Array), "The provided data must be a ctypes or numpy array"
        assert len(field) > 0, "The given array has no size"

    return len(field[0])

## Returns the height of the given field.
# \ingroup SLMDisplayPython
# \param field The field whose height we want.
# \return Zero when there was an error. Otherwise a number greater than zero.
def height(field):
    if supportNumPy and isinstance(field, numpy.ndarray):
        assert field.size > 0, "The given array has no size"

    else:
        assert isinstance(field, ctypes.Array), "The provided data must be a ctypes or numpy array"
        assert len(field) > 0, "The given array has no size"

    return len(field)

## A data handle used when loading data to be shown at a later point.
# \ingroup SLMDisplayPython
class Datahandle(ctypes.Structure):
    ## \cond INTERNALDOC
    ## The fields of the data handle struct.
    ## PYTHONFIELDDATA_BEGIN
    _fields_ = [
        ("id", ctypes.c_uint),
        ("state", ctypes.c_uint8),
        ("errorCode", ctypes.c_uint8),
        ("canvas", ctypes.c_int8),
        ("__padding1", ctypes.c_uint8),
        ("durationInFrames", ctypes.c_uint8),
        ("dataFormat", ctypes.c_uint8),
        ("dataWidth", ctypes.c_ushort),
        ("dataHeight", ctypes.c_ushort),
        ("dataPitch", ctypes.c_ushort),
        ("phaseWrap", ctypes.c_float),
        ("dataFlags", ctypes.c_uint),
        ("transformShiftX", ctypes.c_short),
        ("transformShiftY", ctypes.c_short),
        ("transformScale", ctypes.c_float),
        ("beamSteerX", ctypes.c_float),
        ("beamSteerY", ctypes.c_float),
        ("beamLens", ctypes.c_float),
        ("gamma", ctypes.c_float),
        ("valueOffset", ctypes.c_float),
        ("delayTimeMs", ctypes.c_ushort),
        ("processingWaitTimeMs", ctypes.c_ushort),
        ("loadingTimeMs", ctypes.c_ushort),
        ("conversionTimeMs", ctypes.c_ushort),
        ("processingTimeMs", ctypes.c_ushort),
        ("transferWaitTimeMs", ctypes.c_ushort),
        ("transferTimeMs", ctypes.c_ushort),
        ("renderWaitTimeMs", ctypes.c_ushort),
        ("renderTimeMs", ctypes.c_ushort),
        ("becomeVisibleTimeMs", ctypes.c_ushort),
        ("visibleTimeMs", ctypes.c_ushort),
        ("visibleDurationTimeMs", ctypes.c_ushort),
    ]
    ## PYTHONFIELDDATA_END
    ## \endcond

    ## Represents an action that did not happen yet or was not required.
    NotDone = 0xFFFF

    ## Creates a new data handle for an SLM.
    # \param slm The instance of \ref SLMInstance the handle belongs to.
    def __init__(self, slm):
        assert isinstance(slm, SLMInstance), "Invalid SLM assigned."

        ## Reference to the SLM the datahandle belongs to. It is of the type \ref SLMInstance.
        self.slm = weakref.ref(slm)

        # This only exists for documentation reasons
        if False:
            ## PYTHONFIELDINIT_BEGIN
            ##The id of the data handle. The id distinguishes one handle from another.
            self.id = 0
            ##The state of this data handle. When the state is \ref holoeye.slmdisplaysdk.heds_types.State.Error, then the error is specified in \ref errorCode.
            #See \ref holoeye.slmdisplaysdk.heds_types.State for more info.
            self.state = SLMInstance.State.WaitingForProcessing
            ##The error code of the data handle. The value of \p state will always be \ref holoeye.slmdisplaysdk.heds_types.State.Error when there is an error.
            #See \ref holoeye.slmdisplaysdk.heds_types.ErrorCode for more info.
            self.errorCode = SLMInstance.State.NoError
            ##The canvas on which the data is shown. Only applies to currently unsupported multi-SLM setups.
            self.canvas = -1
            ##For memory alignment.
            self.__padding1 = 0
            ##Specifies for how many graphics video interface frames the data will be visible at least before the next data can go into visible state and replace this data.
            #It specifies the minimum visible duration of this data. The value is based on the refresh rate of the device, i.e. the visible duration in millisenconds can be computed as
            #   visible_duration_ms = DurationInFrames * 1000 / refresh_rate_hz.
            #For instance, setting a duration of 3 frames at 60 Hz refresh rate means it will be visible on the device for at least 50 ms, even if measured timings
            #would report a shorter period due to measurement inaccuracy. The actual visible time may be longer if the next show command is not called early enough,
            #and if it is longer, it is longer by full graphics video interface frames.
            #Default value is 1 (playback at graphics video interface refresh rate), and maximum value is 255. If you need to change data at a lower rate, please use the
            #delay features of your programming language (delay, sleep, pause, ...).
            self.durationInFrames = 1
            ##The format of the data. Refer to \ref holoeye.slmdisplaysdk.heds_types.Format for allowed values.
            self.dataFormat = 255
            ##The number of pixel per line (columns) of the data to show on screen. See \ref DataPitch.
            self.dataWidth = 0
            ##The number of lines (pixel rows) of the data.
            self.dataHeight = 0
            ##The pitch of the data in bytes per line, i.e. number of bytes per row, or per column in case of transposed data.
            #This is the offset between two neighboring lines (columns) in linear memory. Relates to \ref DataFormat.
            #It must be at least \ref DataWidth times number of bytes per pixel, depending on \ref DataFormat, or it can be 0 by
            #default to automatically detect the minimum allowed pitch.
            self.dataPitch = 0
            ##The phase wrap and unit of the provided data. The value represents the angle of a full circle.
            #Default value 0 will make data to be handled as integer or floating point gray values.
            #To interpret floating point data as phase values, please set this property to the unit of your phase data, e.g. 2 pi rad.
            self.phaseWrap = 0.0
            ##Allows to provide load flags and show flags. Load flags define how the data is loaded (e.g. transposed memory layout C/Fortran),
            #and show flags are applied when showing the data on the SLM. Show flags allow applying different presentation modes, which decide
            #how the data is shown on the SLM (centered, resized, flipped, etc.). Show flags are just a display option and do not change the uploaded data.
            self.dataFlags = SLMInstance.ShowFlags.PresentAutomatic
            ##Shifts the data horizontally (in x-direction) by this pixel number while shown on the SLM.
            #This is just a display option and does not change the uploaded data.
            self.transformShiftX = 0
            ##Shifts the data vertically (in y-direction) by this pixel number while shown on the SLM.
            #This is just a display option and does not change the uploaded data.
            self.transformShiftY = 0
            ##A scaling factor of the data size while displayed on the SLM.
            #This is just a display option and does not change the uploaded data.
            self.transformScale = 1.0
            ##A blazed grating is added to the displayed data when this parameter is non-zero. The blazed grating will steer the incident light in x-direction.
            #Please use values in the range [-1.0, 1.0]. For values out of this range the result does not make sense due to the pixel size of the SLM,
            #i.e. for values 1.0 and -1.0 a binary grating is addressed. Please use the function \ref slmdisplaysdk.SLMInstance.utilsBeamSteerFromAngleRad or \ref slmdisplaysdk.SLMInstance.utilsBeamSteerFromAngleDeg 
            #to calculate this parameter out of the deviation angle. The reverse calculation can be done with \ref slmdisplaysdk.SLMInstance.utilsBeamSteerToAngleRad and \ref slmdisplaysdk.SLMInstance.utilsBeamSteerToAngleDeg.
            self.beamSteerX = 0.0
            ##A blazed grating is added to the displayed data when this parameter is not zero. The blazed grating will steer the incident light in y-direction.
            #Please use values in the range [-1.0, 1.0]. For values out of this range the result does not make sense due to the pixel size of the SLM,
            #i.e. for values 1.0 and -1.0 a binary grating is addressed. Please use the function \ref slmdisplaysdk.SLMInstance.utilsBeamSteerFromAngleRad or \ref slmdisplaysdk.SLMInstance.utilsBeamSteerFromAngleDeg 
            #to calculate this parameter out of the deviation angle. The reverse calculation can be done with \ref slmdisplaysdk.SLMInstance.utilsBeamSteerToAngleRad and \ref slmdisplaysdk.SLMInstance.utilsBeamSteerToAngleDeg.
            self.beamSteerY = 0.0
            ##A Fresnel zone lens is added to the displayed data when this parameter is not zero. The Fresnel zone lens will defocus the incident light.
            #The lens power is defined so that values in range [-1.0, 1.0] produces pixel-correct results, and out of this range (e.g. 2.0) Moire-effects occurs,
            #which starts in the SLM-edges and grow with an increasing value. Therefore, depending on your application, values out of the recommended range may still produce
            #a valid lens function, but are in general not pixel-correct. Please use the function \ref slmdisplaysdk.SLMInstance.utilsBeamLensFromFocalLengthMM to calculate this parameter out of the desired
            #focal length. The reverse calculation can be done with \ref slmdisplaysdk.SLMInstance.utilsBeamLensToFocalLengthMM.
            self.beamLens = 0.0
            ##The gamma factor applies a gamma curve on all SLM gray levels while this data is visible on the SLM.
            self.gamma = 1.0
            ##This value adds a constant phase shift to the SLM gray levels while this data is visible on the SLM.
            #The range of this value is from 0 to 1, and one is wrapped back to 0. That means this value is a wrapped phase shift value with phase unit and phase wrap at 1.0.
            #To address a constant gray level offset on the whole SLM of gray level 128 just set this value to 0.5. On a typical 2 pi rad calibrated SLM this would lead to an
            #additional global phase shift value of 1 pi rad or 180 degree.
            self.valueOffset = 0.0
            ##The time in milliseconds between issuing the command to provide and load the data until the process picks up the data for further processing.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.delayTimeMs = NotDone
            ##The time in milliseconds the data had to wait before any processing steps started after loading this data.
            #If no processing is needed, e.g. when uploading data from memory in native formats like 8-bit integer or 32-bit
            #floating point numbers, this value is \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.processingWaitTimeMs = NotDone
            ##The time in milliseconds it took to load data from file.
            #If data was directly provided from memory, i.e. the data is not loaded from file, this value is \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.loadingTimeMs = NotDone
            ##The time in milliseconds it took to convert the data for displaying. If no conversion was required the value is \ref NotDone.
            #Ideally this is always the case since this value represents only data type conversions, which can be avoided by only providing
            #native data formats like 8-bit integer or 32-bit floating point values.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.conversionTimeMs = NotDone
            ##The time in milliseconds it took to process the data. Meaning mathematical operations required to show the data,
            #like processing a color table, changing the pitch if the provided one is not supported natively, applying a
            #phase wrap to non-native data formats, etc. . If no processing was required the value will be \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.processingTimeMs = NotDone
            ##The time in milliseconds the data had to wait after being transferred into graphics card memory.
            #If the data is not ready yet to be transferred, the value will be \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.transferWaitTimeMs = NotDone
            ##The time in milliseconds it took to transfer the data into graphics card memory. If the data has not yet been
            #transferred or an error occured, the value will be \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.transferTimeMs = NotDone
            ##The time in milliseconds the data had to wait before rendering started. If no show function has been called yet for this data,
            #the value will be \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.renderWaitTimeMs = NotDone
            ##The time in milliseconds it took to render the data. This is not the time it was visible but the actual time the graphics card was computing the image output.
            #The data is rendered on each graphics video interface frame as long as the \ref durationInFrames property is not fulfilled.
            #This value always represents the first time the data was rendered. If the data has not been rendered yet or an error occured, the value will be \ref NotDone.
            #The higher this number, the more GPU load is generated. If this value reaches the video interface frame time (typically 16.67 ms for a 60 Hz SLM), the GPU is at
            #full load and the frame rate of data display is about to drop. This will lead to increased visible times of the data handles.
            #Modern gaming GPUs have a lot of computation power and therefore should render very fast compared to typical video interface frame times.
            #When using low-end GPUs or multi GPU environments this value is of more relevance.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.renderTimeMs = NotDone
            ##The time in milliseconds the data had to wait to before the rendered result could actually been made visible on the screen of the SLM window.
            #If the data has not yet been rendered or an error occured, the value will be \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            self.becomeVisibleTimeMs = NotDone
            ##The total time in milliseconds the data was or is still visible on the SLM since its last switch into visible state, until new data was shown or until now, respectively.
            #If the data has not yet been gone into visible state or an error occured, the value will be \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            #The actual visible time is always a positive integer times the video interface frame time. Reported values other than that have measurement errors. If the data went into
            #visible state, the visible time is at least one video frame. If no new data follows after this data, this data will stay visible and the visible time of this data will
            #increase further and further.
            self.visibleTimeMs = NotDone
            ##The time in milliseconds the data was or is still visible on the SLM since its last switch into visible state, until either it reached the state VisibleDurationFinished,
            #or until now if the data is visible and \ref durationInFrames has not passed yet, respectively. The VisibleDurationFinished state is reached after the property \ref durationInFrames
            #has been fulfilled and new data is allowed to be shown, if available. If the data has not yet been shown or an error occured, the value will be \ref NotDone.
            #Please note that time measurement values may have a significant inaccuracy depending on the operating system.
            #This value is smaller than the \ref visibleTimeMs property in case the process calling the show datahandle functions is waiting in between and calls the show function for this handle too late.
            self.visibleDurationTimeMs = NotDone
            ## PYTHONFIELDINIT_END

    ## Destroys the data associated with this handle.
    def __del__(self):
        self.release()

    ## Updates the handle with the latest information.
    # \return ErrorCode.NoError when there is no error.
    def update(self):
        slm = self.slm()

        if slm is not None:
            return slm.updateDatahandle(self)

        return ErrorCode.SLMUnavailable

    ## Releases the data associated with this data handle.
    # \return No return value.
    def release(self):
        if self.id > 0:
            slm = self.slm()

            if slm is not None:
                slm._library.heds_datahandle_release_id(self.id)

                self.id = 0

    ## Ensures that when the datahandle is deleted, the associated data remains valid.
    # You must then use Datahandle.Release(handleid) to release the data.
    # \return The id associated with this handle.
    def retainData(self):
        id = self.id

        ## \cond IGNORE
        self.id = 0
        ## \endcond

        return id

    ## Waits until the data handle has reached a certain state.
    # \param state The state to wait for.
    # \param timeOutInMs The time in milliseconds to wait before returning with the error ErrorCode.WaitForHandleTimedOut.
    # \return ErrorCode.NoError when there is no error.
    def waitFor(self, state, timeOutInMs=4000):
        slm = self.slm()

        if slm is not None:
            return slm.datahandleWaitFor(self, state, timeOutInMs)

        return ErrorCode.SLMUnavailable

## This class gives you access to the SLM Graphics Library and slm devices.
# \ingroup SLMDisplayPython
class SLMInstance:
    ## Creates a new instance of the library.
    # \param binaryFolder The folder where the binaries are stored. Expects a folder that contains "<platform>/holoeye_slmdisplaysdk.dll"
    # \param sharedObject The name of the DLL/shared object to be loaded.
    def __init__(self, binaryFolder = "", sharedObject = "holoeye_slmdisplaysdk"):
        if system() == "Windows":
            sharedObject += ".dll"
            platform = "win32" if sys.maxsize == (2 ** 31 - 1) else "win64"
        else:
            sharedObject = "lib" + sharedObject + ".so"
            platform = "linux"

        if binaryFolder == "":
            binaryFolder = os.getenv("HEDS_3_2_PYTHON", "")

            if binaryFolder != "":
                binaryFolder = os.path.join(binaryFolder, platform)
            else:
                sdklocal = os.path.join(os.getcwd(), platform, sharedObject)
                if os.path.isfile(sdklocal):
                    binaryFolder = os.path.dirname(sdklocal)
                else:
                    sdklocal = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "bin", platform, sharedObject))
                    if os.path.isfile(sdklocal):
                        binaryFolder = os.path.dirname(sdklocal)
                    else:
                        sdklocal = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "bin", platform, sharedObject))
                        if os.path.isfile(sdklocal):
                            binaryFolder = os.path.dirname(sdklocal)

        if binaryFolder == "":
            raise RuntimeError("The environmental variable HEDS_3_2_PYTHON is not set.")

        if not os.path.isdir(binaryFolder):
            raise RuntimeError("The binaries folder \"" + binaryFolder + "\" does not exist.")

        ## The path of the dynamic library file that was loaded.
        self.librarypath = os.path.abspath( os.path.join(binaryFolder, sharedObject) )

        if not os.path.isfile(self.librarypath):
            raise RuntimeError("Cannot find binary file \"" + self.librarypath + "\".")

        try:
            ## \cond INTERNALDOC
            ## The ctypes library object.
            self._library = ctypes.cdll.LoadLibrary(self.librarypath)
            ## \endcond

            # Correct some types
            self._library.heds_error_string_ascii.restype = ctypes.c_char_p
            self._library.heds_info_version_string_ascii.restype = ctypes.c_char_p

            self._library.heds_utils_wait_s.argtypes = (ctypes.c_double,)
            
            self._library.heds_time_now.restype = ctypes.c_uint64
            self._library.heds_time_duration_ms.restype = ctypes.c_double
            self._library.heds_time_duration_ms.argtypes = (ctypes.c_uint64, ctypes.c_uint64,)
            self._library.heds_time_wait_ms.restype = ctypes.c_double
            self._library.heds_time_wait_ms.argtypes = (ctypes.c_double,)

            self._library.heds_utils_beam_steer_to_angle_rad.restype = ctypes.c_float
            self._library.heds_utils_beam_steer_to_angle_deg.restype = ctypes.c_float
            self._library.heds_utils_beam_steer_from_angle_rad.restype = ctypes.c_float
            self._library.heds_utils_beam_steer_from_angle_deg.restype = ctypes.c_float
            self._library.heds_utils_beam_lens_to_focal_length_mm.restype = ctypes.c_float
            self._library.heds_utils_beam_lens_from_focal_length_mm.restype = ctypes.c_float

            class SlmSize(ctypes.Structure):
                _fields_ = [
                    ("width", ctypes.c_int),
                    ("height", ctypes.c_int)
                ]

            self._library.heds_slm_size_px.restype = SlmSize

            class SlmSizeMM(ctypes.Structure):
                _fields_ = [
                    ("width", ctypes.c_double),
                    ("height", ctypes.c_double)
                ]

            self._library.heds_slm_size_mm.restype = SlmSizeMM

            self._library.heds_slm_pixelsize_um.restype = ctypes.c_double
            self._library.heds_slm_pixelsize_m.restype = ctypes.c_double
            self._library.heds_slm_refreshrate_hz.restype = ctypes.c_float

            self._library.heds_slm_width_mm.restype = ctypes.c_double
            self._library.heds_slm_height_mm.restype = ctypes.c_double

            self._library.heds_show_dividedscreen_vertical.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int)
            self._library.heds_show_dividedscreen_horizontal.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int)

            self._library.heds_show_grating_vertical_blaze.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)
            self._library.heds_show_grating_horizontal_blaze.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)

            self._library.heds_show_phasefunction_axicon.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)
            self._library.heds_show_phasefunction_lens.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)
            self._library.heds_show_phasefunction_vortex.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double)

            self._library.heds_show_phasevalues_single.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_uint32, ctypes.c_float)
            self._library.heds_show_phasevalues_double.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_uint32, ctypes.c_float)

            self._library.heds_load_phasevalues_single.argtypes = (ctypes.POINTER(Datahandle), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float, ctypes.c_uint32)
            self._library.heds_load_phasevalues_double.argtypes = (ctypes.POINTER(Datahandle), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_float, ctypes.c_uint32)

            self._library.heds_slm_wavefrontcompensation_load_unicode.argtypes = (ctypes.c_wchar_p, ctypes.c_double, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32)
            self._library.heds_slm_wavefrontcompensation_load_utf8.argtypes = (ctypes.c_char_p, ctypes.c_double, ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32)

        except Exception as ex:
            raise RuntimeError("Failed to initialize library from \"" + self.librarypath + "\". ERROR: " + ex.message)

        ## The version string of the SDK.
        self.version_string = ""

        ## The major version of the SDK. The version format is major.minor.hotfix.revision.
        self.version_major = 0

        ## The minor version of the SDK. The version format is major.minor.hotfix.revision.
        self.version_minor = 0

        ## The hotfix version of the SDK. The version format is major.minor.hotfix.revision.
        self.version_hotfix = 0

        ## The revision version of the SDK. The version format is major.minor.hotfix.revision.
        self.version_revision = 0

        ## The version of the API implemented by the SDK. The API version is not directly related to the SDK version.
        self.version_api = 0

        ## The refreshrate of the slm in Hz.
        self.refreshrate_hz = 0

        ## The pixelsize of the slm in micro-meters.
        self.pixelsize_um = 0

        ## The pixelsize of the slm in meters.
        self.pixelsize_m = 0

        ## The width of the slm in pixel.
        self.width_px = 0

        ## The height of the slm in pixel.
        self.height_px = 0

        ## The size of the slm in pixel.
        self.size_px = 0

        ## The width of the slm in pixel.
        self.width_mm = 0

        ## The height of the slm in pixel.
        self.height_mm = 0

        ## The size of the slm in millimeters.
        self.size_mm = 0

    ## Checks if the used SDK provides the correct API version required by the user's code.
    #  If the function fails, text will be written to the console window. In addition a messagebox can be shown when \p showMessagebox is True.
    # \param requiredApiVersion The API required by your code.
    # \param showMessagebox Defines if a messagebox will be shown, if there is an error.
    # \return Return True when the API version of the SDK is the same as the provided \p requiredApiVersion. */
    def requiresVersion(self, requiredApiVersion, showMessagebox = False):
        return bool(self._library.heds_requires_version(ctypes.c_int(requiredApiVersion), ctypes.c_int(showMessagebox)))

    ## \cond INTERNALDOC
    ## Checks if the given data is a data handle id.
    # \return True when \p data is a data handle id.
    @staticmethod
    def _isID(data):
        return isinstance(data, int) or (not isPython3 and isinstance(data, long)) or isinstance(data, ctypes.c_int) or isinstance(data, ctypes.c_uint)
    ## \endcond

    ## Provides an error string for a given error.
    # \param error The error code to provide the string for.
    # \return An error string for the given error. None when the given error is invalid.
    def errorString(self, error):
        return self._library.heds_error_string_ascii(error)

    ## Opens the slm window when not already open. Must be called before using most functions of SLM Display SDK.
    # \return ErrorCode.NoError when there is no error.
    def open(self):
        apinamesize = 32
        slmsize = 32

        class Configuration(ctypes.Structure):
            _fields_ = [
                ("api", ctypes.c_int),
                ("apiname", ctypes.c_char * apinamesize),
                ("slm", ctypes.c_char * slmsize),
            ]

        def fixed(str, size):
            size -= 1
            if len(str) > size:
                str = str[:size]

            return str.encode("ascii")

        config = Configuration()
        config.api = 3
        config.apiname = fixed("{}.{}.{} {} {}".format(*sys.version_info), apinamesize)

        err = self._library.heds_slm_init( ctypes.pointer(config) )

        if err != ErrorCode.NoError:
            return err

        self.version_string = self._library.heds_info_version_string_ascii()
        self.version_major = self._library.heds_info_version_major()
        self.version_minor = self._library.heds_info_version_minor()
        self.version_hotfix = self._library.heds_info_version_hotfix()
        self.version_revision = self._library.heds_info_version_revision()
        self.version_api = self._library.heds_info_version_api()

        self.refreshrate_hz = self._library.heds_slm_refreshrate_hz()
        self.pixelsize_um = self._library.heds_slm_pixelsize_um()
        self.pixelsize_m = self.pixelsize_um / 1.0E6

        size = self._library.heds_slm_size_px()

        self.width_px = size.width
        self.height_px = size.height
        self.size_px = (self.width_px, self.height_px)
        self.width_mm = self.width_px * self.pixelsize_m * 1000
        self.height_mm = self.height_px * self.pixelsize_m * 1000
        self.size_mm = (self.width_mm, self.height_mm)

        return ErrorCode.NoError

    ## Deletes the instance and closes the SLM window.
    def __del__(self):
        self._library.heds_slm_close()

    ## Closes the slm window when open.
    # \return No return type.
    def close(self):
        self._library.heds_slm_close()

        self.refreshrate_hz = 0
        self.pixelsize_um = 0
        self.pixelsize_m = 0
        self.width_px = 0
        self.height_px = 0
        self.size_px = 0
        self.width_mm = 0
        self.height_mm = 0
        self.size_mm = 0

    ## Show or hide the SLM preview window. SLM must be initialized before opening the preview window.
    # \param show When True the window will be shown. When False it will be closed.
    # \return ErrorCode.NoError when there is no error.
    def utilsSLMPreviewShow(self, show = True):
        return int(self._library.heds_utils_slmpreview_show(ctypes.c_int(show)))

    ## Sets settings and the zoom factor of the preview window.
    # \param flags The preview flags to set. Refer to \ref holoeye.slmdisplaysdk.heds_types.SLMPreviewFlags for details.
    # \param zoom The zoom factor of the preview window. Use zero to make the data fit the screen.
    # \return ErrorCode.NoError when there is no error.
    def utilsSLMPreviewSet(self, flags, zoom):
        return int(self._library.heds_utils_slmpreview_set(ctypes.c_int(flags), ctypes.c_float(zoom)))

    ## Changes the position and size of the preview window.
    # \param posX The horizontal position of the window on the desktop.
    # \param posY The vertical position of the window on the desktop.
    # \param width The width of the window. If \p width or \p height is zero, the size will not be changed.
    # \param height The height of the window. If \p width or \p height is zero, the size will not be changed.
    # \return ErrorCode.NoError when there is no error.
    def utilsSLMPreviewMove(self, posX, posY, width = 0, height = 0):
        return int(self._library.heds_utils_slmpreview_move(ctypes.c_int(posX), ctypes.c_int(posY), ctypes.c_int(width), ctypes.c_int(height)))

    ## Waits for a given amount of time.
    # \param millisecondsToWait The number of milliseconds to wait.
    # \return No return value.
    def utilsWaitForMs(self, millisecondsToWait):
        time.sleep(millisecondsToWait / 1000.0)

    ## Waits for a given amount of time.
    # \param secondsToWait The number of seconds to wait.
    # \return No return value.
    def utilsWaitForS(self, secondsToWait):
        time.sleep(secondsToWait)

    ## Waits for a given amount of time. Ends when the SLM process is closed.
    # \param millisecondsToWait The number of milliseconds to wait.
    # \return ErrorCode.NoError when the desired wait time was reached.
    def utilsWaitForCheckedMs(self, millisecondsToWait):
        return self._library.heds_utils_wait_checked_ms(ctypes.c_int(millisecondsToWait))

    ## Waits for a given amount of time. Ends when the SLM process is closed.
    # \param secondsToWait The number of seconds to wait.
    # \return ErrorCode.NoError when the desired wait time was reached.
    def utilsWaitForCheckedS(self, secondsToWait):
        return self._library.heds_utils_wait_checked_s(ctypes.c_double(secondsToWait))

    ## Waits until SLM process is closed.
    # \return Always returns an error.
    def utilsWaitUntilClosed(self):
        return self._library.heds_utils_wait_until_closed()
       
    ## Get a precise time point in platform dependent unit. Please use \ref heds_time_duration_ms()
    ## to measure a duration between two time points.
    ## Is meant to be used in languages with less accurate time measurement implementations.
    # \return An arbitrary unsigned 64-bit integer value representing the current time. */
    def timeNow(self):
        return self._library.heds_time_now()
       
    ## Get the duration between two calls to \ref heds_time_now() in floating point milliseconds to not loose microsendond precision, if available.
    # \param t2 The end time point of the duration to be measured. Must be in unit returned by \ref heds_time_now().
    # \param t1 The start time point of the duration to be measured. Must be in unit returned by \ref heds_time_now().
    # \return The duration between both time points in floating point milliseconds. */
    def timeDurationMs(self, t2, t1):
        return self._library.heds_time_duration_ms(ctypes.c_uint64(t2), ctypes.c_uint64(t1));
       
    ## A precise waiting function compared to utilsWaitFor... functions. Compared to the utils wait functions,
    ## this function will block the current thread with full load on one CPU core while waiting.
    ## Uses \ref heds_time_now() and \ref heds_time_duration_ms() functions to measure the time internally.
    ## It is recommended to use this function for short but precise durations, like below 100 ms.
    ## In case of waiting for longer periods, there is the function \ref heds_utils_wait_checked_ms(), which will stop
    ## waiting in case the process is closed in between.
    # \param min_wait_dur_ms The duration to block the current thread for in floating point milliseconds.
    #                        This duration is the minimum executation duration of this function. There is no guaranty
    #                        the operating system will allow this function to return at the end of the period, but typically
    #                        it returns within a few microseconds after the period has passed. Please check the return value
    #                        for the actual execution time.
    # \return The actual measured executation duration of this function call in floating point milliseconds. */
    def timeWaitMs(self, millisecondsToWait):
        return self._library.heds_time_wait_ms(ctypes.c_double(millisecondsToWait))

    ## Utilities function for calculating proper values for the beam manipulation parameters in Datahandle.
    # The function takes the beam steer value from the data handle property as an input and calculates the corresponding
    # steering angle of the incident light in radian.
    # The SLM must be initialized properly in order to return the correct value.
    # In case of an error due to uninitialized SLM the function returns 0.0.
    # \param wavelength_nm The wavelength of the incident light in SI-unit nanometer.
    # \param beam_steer The parameter passed to the data handle property "beamSteerX" or "beamSteerY". Best optical blaze results are gained for values between -1.0 and 1.0.
    # \return Returns the corresponding deviation angle in radian (full circle is 2*pi rad).
    def utilsBeamSteerToAngleRad(self, wavelength_nm, beam_steer):
        return float(self._library.heds_utils_beam_steer_to_angle_rad(ctypes.c_float(wavelength_nm), ctypes.c_float(beam_steer)))

    ## Utilities function for calculating proper values for the beam manipulation parameters in Datahandle.
    # The function takes the beam steer value from the datahandle property as an input and calculates the corresponding
    # steering angle of the incident light in degree.
    # The SLM must be initialized properly in order to return the correct value.
    # In case of an error due to uninitialized SLM the function returns 0.0f.
    # \param wavelength_nm The wavelength of the incident light in SI-unit nanometer.
    # \param beam_steer The parameter passed to the datahandle property "beamSteerX" or "beamSteerY". Best optical blaze results are gained for values between -1.0 and 1.0.
    # \return Returns the corresponding deviation angle in degree (full circle is 360 degree).
    def utilsBeamSteerToAngleDeg(self, wavelength_nm, beam_steer):
        return float(self._library.heds_utils_beam_steer_to_angle_deg(ctypes.c_float(wavelength_nm), ctypes.c_float(beam_steer)))

    ## Utilities function for calculating proper values for the beam manipulation parameters in Datahandle.
    # The function takes the desired steering angle of the incident light in radian as an input and calculates the
    # corresponding beam steer parameter to be passed into Datahandle. The beam steer parameter is normalized to
    # meaningful values in the range from -1.0 to +1.0. The value corresponds to steering from one side of the
    # unit cell to the other side in the far field of a holographic projection.
    # The SLM must be initialized properly in order to return the correct value.
    # In case of an error due to uninitialized SLM the function returns 0.0.
    # \param wavelength_nm The wavelength of the incident light in SI-unit nanometer.
    # \param steering_angle_rad Desired steering angle of the incident light in radian (full circle is 2*pi rad).
    # \return Returns the corresponding beam steer parameter to be passed into Datahandle. Values in range [-1.0, 1.0] are recommended.
    def utilsBeamSteerFromAngleRad(self, wavelength_nm, steering_angle_rad):
        return float(self._library.heds_utils_beam_steer_from_angle_rad(ctypes.c_float(wavelength_nm), ctypes.c_float(steering_angle_rad)))

    ## Utilities function for calculating proper values for the beam manipulation parameters in Datahandle.
    # The function takes the desired steering angle of the incident light in degree as an input and calculates the
    # corresponding beam steer parameter to be passed into Datahandle. The beam steer parameter is normalized to
    # meaningful values in the range from -1.0 to +1.0. The value corresponds to steering from one side of the
    # unit cell to the other side in the far field of a holographic projection.
    # The SLM must be initialized properly in order to return the correct value.
    # In case of an error due to uninitialized SLM the function returns 0.0.
    # \param wavelength_nm The wavelength of the incident light in SI-unit nanometer.
    # \param steering_angle_deg Desired steering angle of the incident light in degree (full circle is 360 degree).
    # \return Returns the corresponding beam steer parameter to be passed into Datahandle. Values in range [-1.0, 1.0] are recommended.
    def utilsBeamSteerFromAngleDeg(self, wavelength_nm, steering_angle_deg):
        return float(self._library.heds_utils_beam_steer_from_angle_deg(ctypes.c_float(wavelength_nm), ctypes.c_float(steering_angle_deg)))

    ## Utilities function for calculating proper values for the beam manipulation parameters in Datahandle.
    # The function takes the beam lens parameter value from the datahandle property as an input and calculates
    # the corresponding focal length of the Fresnel zone lens addressed with the given beam lens parameter.
    # The beam lens parameter is proportional to the lens power (1/f) and is scaled so that for values in the range
    # between -1.0 and +1.0 the addressed phase function has no artifacts due to the pixel size of the SLM.
    # Higher absolute values might still produce valid optical lens results, but the quality of the addressed lens
    # phase function will degrade with an increasing absolute value above 1.0.
    # The SLM must be initialized properly in order to return the correct value.
    # In case of an error due to uninitialized SLM the function returns 0.0.
    # \param wavelength_nm The wavelength of the incident light in SI-unit nanometer.
    # \param beam_lens The parameter passed to the datahandle property "beamLens". Values in range [-1.0, 1.0] are recommended.
    # \return Returns the corresponding focal length of the Fresnel zone lens in SI-unit mm.
    def utilsBeamLensToFocalLengthMM(self, wavelength_nm, beam_lens):
        return float(self._library.heds_utils_beam_lens_to_focal_length_mm(ctypes.c_float(wavelength_nm), ctypes.c_float(beam_lens)))

    ## Utilities function for calculating proper values for the beam manipulation parameters in Datahandle.
    # The function takes the desired folcal length as an input and calculates the corresponding beam lens parameter.
    # The beam lens parameter is proportional to the lens power (1/f) and is scaled so that for values in the range
    # between -1.0 and +1.0 the addressed phase function has no artifacts due to the pixel size of the SLM.
    # Higher absolute values might still produce valid optical lens results, but the quality of the addressed lens
    # phase function will degrade with an increasing absolute value above 1.0.
    # The SLM must be initialized properly in order to return the correct value.
    # In case of an error due to uninitialized SLM the function returns 0.0.
    # \param wavelength_nm The wavelength of the incident light in SI-unit nanometer.
    # \param focal_length_mm Desired focal length in SI-unit millimeter.
    # \return Returns the corresponding "beamLens" parameter to be passed into Datahandle. Values in range [-1.0, 1.0] are recommended.
    def utilsBeamLensFromFocalLengthMM(self, wavelength_nm, focal_length_mm):
        return float(self._library.heds_utils_beam_lens_from_focal_length_mm(ctypes.c_float(wavelength_nm), ctypes.c_float(focal_length_mm)))

    ## Shows a given image file on the slm.
    # \param imageFilePath A string containing the path to an image file.
    # <br>Supported image file formats are: bmp, cur, dds, gif, icns, ico, jpeg, jpg, pbm, pgm, png, ppm, svg, svgz, tga, tif, tiff, wbmp, webp, xbm, xpm.
    # <br>For holographic data, we recommend not to use any lossy compressed formats, like jpg. Instead please use uncompressed formats (e.g. bmp) or lossless compressed formats (e.g. png).
    # \param showFlags Flags that define how the data is shown on the slm.
    # \return ErrorCode.NoError when there is no error.
    # \see \ref imagefile_grayscale.py \ref imagefile_rgb.py
    def showDataFromFile(self, imageFilePath, showFlags = 0):
        abspath = os.path.abspath(imageFilePath)

        isUnicode = (isinstance(abspath, str)) if isPython3 else (isinstance(abspath, unicode))

        if isUnicode:
            if not os.path.exists(abspath):
                return ErrorCode.FileNotFound

            return self._library.heds_show_data_fromfile_unicode(abspath, showFlags)

        # we do not check if the file exists here because this would fail for utf-8 strings
        return self._library.heds_show_data_fromfile_utf8(abspath, showFlags)

    ## Shows data associated with a handle on the slm.
    # \param handleOrID A datahandle or a datahandle id.
    # \param showFlags Flags that define how the data is shown on the slm.
    # \return ErrorCode.NoError when there is no error.
    # \see \ref slideshow_data_preload.py
    def showDatahandle(self, handleOrID, showFlags=0):
        if isinstance(handleOrID, Datahandle):
            handleptr = ctypes.pointer(handleOrID)

            return self._library.heds_show_datahandle(handleptr, showFlags)

        if SLMInstance._isID(handleOrID):
            return self._library.heds_show_datahandle_id(handleOrID, showFlags)

        return ErrorCode.InvalidHandle

    ## Shows arbitrary data on the slm.
    # \param data A ctypes or numpy data pointer.
    # \param showFlags Flags that define how the data is shown on the slm.
    # \return ErrorCode.NoError when there is no error.
    # \see \ref data_uint8.py \ref data_uint8_tiled.py \ref data_float.py \ref data_float_tiled.py \ref slideshow_data_show.py \ref imagefile_rgb_data.py
    def showData(self, data, showFlags = 0):
        if supportNumPy and isinstance(data, numpy.ndarray):
            height = len(data)

            if height < 1:
                return ErrorCode.InvalidDataHeight

            width = len(data[0])

            if width < 1:
                return ErrorCode.InvalidDataWidth

            type = data.dtype.type

            if type is numpy.uint8:
                if len(data.shape) < 3:
                    return self._library.heds_show_data_grayscale_uchar(width, height, data.ctypes, 0, showFlags)
                if len(data.shape) == 3:
                    if (data.shape[2] == 1):
                        return self._library.heds_show_data_grayscale_uchar(width, height, data.ctypes, 0, showFlags)
                    elif (data.shape[2] == 3):
                        return self._library.heds_show_data_rgb24(width, height, data.ctypes, 0, showFlags)
                    elif (data.shape[2] == 4):
                        return self._library.heds_show_data_rgba32(width, height, data.ctypes, 0, showFlags)

            if type is numpy.single:
                return self._library.heds_show_data_grayscale_single(width, height, data.ctypes, 0,  showFlags)

            if type is numpy.double:
                return self._library.heds_show_data_grayscale_double(width, height, data.ctypes, 0,  showFlags)

        elif isinstance(data, ctypes.Array):
            height = len(data)

            if height < 1:
                return ErrorCode.InvalidDataHeight

            width = len(data[0])

            if width < 1:
                return ErrorCode.InvalidDataWidth

            type = data._type_._type_

            if type is ctypes.c_ubyte:
                return self._library.heds_show_data_grayscale_uchar(width, height, data, 0,  showFlags)

            if type is ctypes.c_float:
                return self._library.heds_show_data_grayscale_single(width, height, data, 0, showFlags)

            if type is ctypes.c_double:
                return self._library.heds_show_data_grayscale_double(width, height, data, 0, showFlags)

        return ErrorCode.InvalidDataFormat

    ## Shows an array of phase values on the SLM. The unit of the phase values is the same as for \p phaseWrap. By default radians.
    # \param phaseValues The pointer to the given phase values. The unit of the phase values is radian.
    # \param showFlags Flags that define how the data is shown on the slm.
    # \param phaseWrap The phase wrap applied to the data, basically a modulo. A value of zero means there is no phase wrap applied.
    # \return ErrorCode.NoError when there is no error.
    # \see \ref phasevalues.py \ref phasevalues_tiled.py \ref axicon.py \see \ref axicon_fast.py
    def showPhasevalues(self, phaseValues, showFlags = 0, phaseWrap = 6.28318530718):
        if supportNumPy and isinstance(phaseValues, numpy.ndarray):
            height = len(phaseValues)

            if height < 1:
                return ErrorCode.InvalidDataHeight

            width = len(phaseValues[0])

            if width < 1:
                return ErrorCode.InvalidDataWidth

            type = phaseValues.dtype.type

            if type is numpy.single:
                return self._library.heds_show_phasevalues_single(width, height, ctypes.cast(phaseValues.ctypes, ctypes.POINTER(ctypes.c_float)), 0, showFlags, phaseWrap)

            if type is numpy.double:
                return self._library.heds_show_phasevalues_double(width, height, ctypes.cast(phaseValues.ctypes, ctypes.POINTER(ctypes.c_double)), 0, showFlags, phaseWrap)

        elif isinstance(phaseValues, ctypes.Array):
            height = len(phaseValues)

            if height < 1:
                return ErrorCode.InvalidDataHeight

            width = len(phaseValues[0])

            if width < 1:
                return ErrorCode.InvalidDataWidth

            type = phaseValues._type_._type_

            if type is ctypes.c_float:
                return self._library.heds_show_phasevalues_single(width, height, ctypes.cast(phaseValues, ctypes.POINTER(ctypes.c_float)), 0, showFlags, phaseWrap)

            if type is ctypes.c_double:
                return self._library.heds_show_phasevalues_double(width, height, ctypes.cast(phaseValues, ctypes.POINTER(ctypes.c_double)), 0, showFlags, phaseWrap)

        return ErrorCode.InvalidDataFormat

    ## Shows a blank screen with a constant gray value.
    # \param grayValue The gray value which is addressed to the full SLM. The value is automatically wrapped to the range 0-255.
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_blankscreen.py
    def showBlankscreen(self, grayValue = 128):
        return self._library.heds_show_blankscreen(grayValue)

    ## Shows two areas on the SLM with two different gray values. The function is intended to be used for phase measurements of the SLM in which one half of the SLM can be used as a reference to the other half.
    # The screen will be split along the vertical (y) axis. This means that the gray levels a and b are painted to the left and right side of the SLM, resp.
    # \param screenDivider The ratio by which the SLM screen should be divided. Meaningful values are between 0.0 and 1.0. [default value = 0.5]
    # \param a_gray_value The gray value which will be adressed on the first side of the SLM. Values are wrapped to 0-255 range. [default value = 0]
    # \param b_gray_value The gray value which will be adressed on the second side of the SLM. Values are wrapped to 0-255 range. [default value = 255]
    # \param flipped If set to true, the first side will addressed with \p b_gray_value and the second side will be set to a_gray_value. [default value = false]
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_dividedscreen.py
    def showDividedScreenVertical(self, a_gray_value = 0, b_gray_value = 255, screenDivider = 0.5, flipped = False):
        return self._library.heds_show_dividedscreen_vertical(a_gray_value, b_gray_value, screenDivider, flipped)

    ## Shows two areas on the SLM with two different gray values. The function is intended to be used for phase measurements of the SLM in which one half of the SLM can be used as a reference to the other half.
    # The screen will be split along the horizontal (x) axis. This means that the gray levels a and b are painted to the upper and lower side of the SLM, resp.
    # \param screenDivider The ratio by which the SLM screen should be divided. Meaningful values are between 0.0 and 1.0. [default value = 0.5]
    # \param a_gray_value The gray value which will be adressed on the first side of the SLM. Values are wrapped to 0-255 range. [default value = 0]
    # \param b_gray_value The gray value which will be adressed on the second side of the SLM. Values are wrapped to 0-255 range. [default value = 255]
    # \param flipped If set to true, the first side will addressed with \p b_gray_value and the second side will be set to a_gray_value. [default value = false]
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_dividedscreen.py
    def showDividedScreenHorizontal(self, a_gray_value = 0, b_gray_value = 255, screenDivider = 0.5, flipped = False):
        return self._library.heds_show_dividedscreen_horizontal(a_gray_value, b_gray_value, screenDivider, flipped)

    ## Shows a vertical binary grating. The grating consists of two gray values \p a_gray_value and \p b_gray_value which will be addressed to the SLM.
    # The gray values have the data type int and will be wrapped internally to an unsigned char with a range of 0 to 255.
    # The width of each area with the gray value \p a_gray_value and \p b_gray_value is defined by \p a_width and \p b_width, respectively.
    # Each pair of gray values is repeated so that the SLM is completely filled.
    # \param a_width The width of the first block with the value of \p a_gray_value. This parameter is mandatory.
    # \param b_width The width of the second block with the value of \p b_gray_value. This parameter is mandatory.
    # \param a_gray_value The addressed gray value of the first block. [default value = 0].
    # \param b_gray_value The addressed gray value of the second block. [default value = 128].
    # \param shift_x The horizontal offset applied to both blocks. [default value = 0].
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_grating_binary.py
    def showGratingVerticalBinary(self, a_width, b_width, a_gray_value = 0, b_gray_value = 128, shift_x = 0):
        return self._library.heds_show_grating_vertical_binary(a_width, b_width, a_gray_value, b_gray_value, shift_x)

    ## Shows a horizontal binary grating. The grating consists of two gray values \p a_gray_value and \p b_gray_value.
    # The gray values have the data type int and will be wrapped internally to an unsigned char with a range of 0 to 255.
    # Each pair of gray values is repeated so that the SLM is completely filled.
    # The width of each area with the gray value \p a_gray_value and \p b_gray_value is defined by \p a_width and \p b_width, respectively.
    # \param a_width The width of the first block with the value of \p a_gray_value. This parameter is mandatory.
    # \param b_width The width of the second block with the value of \p b_gray_value. This parameter is mandatory.
    # \param a_gray_value The addressed gray value of the first block. [default value = 0].
    # \param b_gray_value The addressed gray value of the second block. [default value = 128].
    # \param shift_y The vertical offset applied to both blocks. [default value = 0].
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_grating_binary.py
    def showGratingHorizontalBinary(self, a_width, b_width, a_gray_value = 0, b_gray_value = 128, shift_y = 0):
        return self._library.heds_show_grating_horizontal_binary(a_width, b_width, a_gray_value, b_gray_value, shift_y)

    ## Shows a vertical blazed grating on the SLM.
    # \param period The grating period in SLM pixels. The value is mandatory. Can be either positive or negative for an inverted grating profile. Please note that the phase can also be inverted by the \p phase_scale. If both values are negative, the invertions will superimpose to non invertion.
    # \param shift_x The horizontal offset applied to the grating. [default value = 0].
    # \param phase_scale Scales all phase values of this phase function. The value can be negative to invert the phase function. Other values than 1.0 and -1.0 are meant to optimize diffraction efficiency. Absolute values greater than 1.0 would lead to gray level saturation artifacts and are therefore limited to the range from -1.0 to +1.0. In case of limitation, a warning message will be shown. The scaling is done after wrapping phase values into the gray levels of the SLM. [default value = 1.0].
    # \param phase_offset Applies an offset to the phase values of this phase function. The unit of this value is in radian. The offset will retain the phase profile, but will change the actual used gray levels on the SLM. When this value is 0, the phase function will be centered into the gray value range on the SLM. After the offset was applied, wrapping to the gray values is done. [default value = 0.0].
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_grating_blaze.py
    def showGratingVerticalBlaze(self, period, shift_x = 0, phase_scale = 1.0, phase_offset = 0.0):
        return self._library.heds_show_grating_vertical_blaze(period, shift_x, phase_scale, phase_offset)

    ## Shows a horizontal blazed grating on the SLM.
    # \param period The grating period in SLM pixels. The value is mandatory. Can be either positive or negative for an inverted grating profile.
    # \param shift_y The vertical offset applied to the grating. [default value = 0].
    # \param phase_scale Scales all phase values of this phase function. The value can be negative to invert the phase function. Other values than 1.0 and -1.0 are meant to optimize diffraction efficiency. Absolute values greater than 1.0 would lead to gray level saturation artifacts and are therefore limited to the range from -1.0 to +1.0. In case of limitation, a warning message will be shown. The scaling is done after wrapping phase values into the gray levels of the SLM. [default value = 1.0].
    # \param phase_offset Applies an offset to the phase values of this phase function. The unit of this value is in radian. The offset will retain the phase profile, but will change the actual used gray levels on the SLM. When this value is 0, the phase function will be centered into the gray value range on the SLM. After the offset was applied, wrapping to the gray values is done. [default value = 0.0].
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_grating_blaze.py
    def showGratingHorizontalBlaze(self, period, shift_y = 0, phase_scale = 1.0, phase_offset = 0.0):
        return self._library.heds_show_grating_horizontal_blaze(period, shift_y, phase_scale, phase_offset)

    ## Shows an axicon. The phase has a conical shape.
    # \param inner_radius_px The radius in number of SLM pixel where the axicon phase function reached 2pi for the first time in respect to the center of the axicon.
    # \param center_x Horizontal shift of the center of the optical function on the SLM in number of pixel. [default value = 0].
    # \param center_y Vertical shift of the center of the optical function on the SLM in number of pixel. [default value = 0].
    # \param phase_scale Scales all phase values of this phase function. The value can be negative to invert the phase function. Other values than 1.0 and -1.0 are meant to optimize diffraction efficiency. Absolute values greater than 1.0 would lead to gray level saturation artifacts and are therefore limited to the range from -1.0 to +1.0. In case of limitation, a warning message will be shown. The scaling is done after wrapping phase values into the gray levels of the SLM. [default value = 1.0].
    # \param phase_offset Applies an offset to the phase values of this phase function. The unit of this value is in radian. The offset will retain the phase profile, but will change the actual used gray levels on the SLM. When this value is 0, the phase function will be centered into the gray value range on the SLM. After the offset was applied, wrapping to the gray values is done. [default value = 0.0].
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_axicon.py
    def showPhasefunctionAxicon(self, inner_radius_px, center_x = 0, center_y = 0, phase_scale = 1.0, phase_offset = 0.0):
        return self._library.heds_show_phasefunction_axicon(inner_radius_px, center_x, center_y, phase_scale, phase_offset)

    ## Shows a lens phase function. The phase has a parabolic shape.
    # The resulting focal length can be calculated as f [m] = (\p inner_radius_px * \p pixelsize_um * 1.0E-6) ^2 / (2.0*lambda),
    # with the incident optical wavelength lambda.
    # \param inner_radius_px The radius in number of SLM pixel where the lens phase function reached 2pi for the first time in respect to the center of the lens. This value is related to the focal length f of the lens phase function by f = (inner_radius_px * heds_slm_pixelsize())^2 / (2*lambda).
    # \param center_x Horizontal shift of the center of the optical function on the SLM in number of pixel. [default value = 0].
    # \param center_y Vertical shift of the center of the optical function on the SLM in number of pixel. [default value = 0].
    # \param phase_scale Scales all phase values of this phase function. The value can be negative to invert the phase function. Other values than 1.0 and -1.0 are meant to optimize diffraction efficiency. Absolute values greater than 1.0 would lead to gray level saturation artifacts and are therefore limited to the range from -1.0 to +1.0. In case of limitation, a warning message will be shown. The scaling is done after wrapping phase values into the gray levels of the SLM. [default value = 1.0].
    # \param phase_offset Applies an offset to the phase values of this phase function. The unit of this value is in radian. The offset will retain the phase profile, but will change the actual used gray levels on the SLM. When this value is 0, the phase function will be centered into the gray value range on the SLM. After the offset was applied, wrapping to the gray values is done. [default value = 0.0].
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_lens.py
    def showPhasefunctionLens(self, inner_radius_px, center_x = 0, center_y = 0, phase_scale = 1.0, phase_offset = 0.0):
        return self._library.heds_show_phasefunction_lens(inner_radius_px, center_x, center_y, phase_scale, phase_offset)

    ## Shows an optical vortex phase function on the SLM. The phase has a helical shape.
    # \param vortex_order The order of the optical vortex. If the order is one, the phase goes from 0 to 2pi over the full angle of 360 degree. For higher orders, 2pi phase shift is reached at angles of 360 degree divided by the given \p vortex_order. [default value = 1].
    # \param inner_radius_px The radius at the sigularity which will be set to gray value 0 on the SLM. [default value = 0].
    # \param center_x Horizontal shift of the center of the optical function on the SLM in number of pixel. [default value = 0].
    # \param center_y Vertical shift of the center of the optical function on the SLM in number of pixel. [default value = 0].
    # \param phase_scale Scales all phase values of this phase function. The value can be negative to invert the phase function. Other values than 1.0 and -1.0 are meant to optimize diffraction efficiency. Absolute values greater than 1.0 would lead to gray level saturation artifacts and are therefore limited to the range from -1.0 to +1.0. In case of limitation, a warning message will be shown. The scaling is done after wrapping phase values into the gray levels of the SLM. [default value = 1.0].
    # \param phase_offset Applies an offset to the phase values of this phase function. The unit of this value is in radian. The offset will retain the phase profile, but will change the actual used gray levels on the SLM. When this value is 0, the phase function will be centered into the gray value range on the SLM. After the offset was applied, wrapping to the gray values is done. [default value = 0.0].
    # \return ErrorCode.NoError when there is no error.
    # \see \ref builtin_vortex.py
    def showPhasefunctionVortex(self, vortex_order = 1, inner_radius_px = 0, center_x = 0, center_y = 0, phase_scale = 1.0, phase_offset = 0.0):
        return self._library.heds_show_phasefunction_vortex(vortex_order, inner_radius_px, center_x, center_y, phase_scale, phase_offset)

    ## \cond INTERNALDOC
    ## Generates an error with a handle
    def _handleError(self, error):
        h = Datahandle(self)
        h.state = State.Error;
        h.errorCode = error;

        return (error, h)
    ## \endcond

    ## Loads data to be displayed later.
    # \param data The data you want to load for the slm.
    # \param loadFlags The flags used when loading the data.
    # \return A tuple (errorCode, handle). You can check if errorCode is ErrorCode.NoError.
    # \see \ref slideshow_data_preload.py
    def loadData(self, data, loadFlags = LoadFlags.Default):
        handle = Datahandle(self)
        handleptr = ctypes.pointer(handle)

        if supportNumPy and isinstance(data, numpy.ndarray):
            height = len(data)

            if height < 1:
                return self._handleError(ErrorCode.InvalidDataHeight)

            width = len(data[0])

            if width < 1:
                return self._handleError(ErrorCode.InvalidDataWidth)

            type = data.dtype.type

            if type is numpy.uint8:
                if len(data.shape) < 3:
                    error = self._library.heds_load_data_grayscale_uchar(handleptr, width, height, data.ctypes, 0, loadFlags)
                    return (error, handle)
                if len(data.shape) == 3:
                    if (data.shape[2] == 1):
                        error = self._library.heds_load_data_grayscale_uchar(handleptr, width, height, data.ctypes, 0, loadFlags)
                        return (error, handle)
                    elif (data.shape[2] == 3):
                        error = self._library.heds_load_data_rgb24(handleptr, width, height, data.ctypes, 0, loadFlags)
                        return (error, handle)
                    elif (data.shape[2] == 4):
                        error = self._library.heds_load_data_rgba32(handleptr, width, height, data.ctypes, 0, loadFlags)
                        return (error, handle)

            if type is numpy.single:
                error = self._library.heds_load_data_grayscale_single(handleptr, width, height, data.ctypes, 0, loadFlags)
                return (error, handle)

            if type is numpy.double:
                error = self._library.heds_load_data_grayscale_double(handleptr, width, height, data.ctypes, 0, loadFlags)
                return (error, handle)

        elif isinstance(data, ctypes.Array):
            height = len(data)

            if height < 1:
                return self._handleError(ErrorCode.InvalidDataHeight)

            width = len(data[0])

            if width < 1:
                return self._handleError(ErrorCode.InvalidDataWidth)

            type = data._type_._type_

            if type is ctypes.c_ubyte:
                error = self._library.heds_load_data_grayscale_uchar(handleptr, width, height, data, 0)
                return (error, handle)

            if type is ctypes.c_float:
                error = self._library.heds_load_data_grayscale_single(handleptr, width, height, data, 0)
                return (error, handle)

            if type is ctypes.c_double:
                error = self._library.heds_load_data_grayscale_double(handleptr, width, height, data, 0)
                return (error, handle)

        return self._handleError(ErrorCode.UnsupportedDataFormat)

    ## Loads the given phase values to be shown on the SLM.
    # \param phaseValues The pointer to the given phase values.
    # \param phaseWrap The phase shift applied to the data. A value of zero means there is no phase shift applied.
    # \param loadFlags The flags used when loading the data.
    # \return A tuple (errorCode, handle). You can check if errorCode is ErrorCode.NoError.
    def loadPhasevalues(self, phaseValues, phaseWrap = 6.28318530718, loadFlags = LoadFlags.Default):
        handle = Datahandle(self)
        handleptr = ctypes.pointer(handle)

        if supportNumPy and isinstance(phaseValues, numpy.ndarray):
            height = len(phaseValues)

            if height < 1:
                return self._handleError(ErrorCode.InvalidDataHeight)

            width = len(phaseValues[0])

            if width < 1:
                return self._handleError(ErrorCode.InvalidDataWidth)

            type = phaseValues.dtype.type

            if type is numpy.single:
                error = self._library.heds_load_phasevalues_single(handleptr, width, height, ctypes.cast(phaseValues.ctypes, ctypes.POINTER(ctypes.c_float)), 0, phaseWrap, loadFlags)
                return (error, handle)

            if type is numpy.double:
                error = self._library.heds_load_phasevalues_double(handleptr, width, height, ctypes.cast(phaseValues.ctypes, ctypes.POINTER(ctypes.c_double)), 0, phaseWrap, loadFlags)
                return (error, handle)

        elif isinstance(phaseValues, ctypes.Array):
            height = len(phaseValues)

            if height < 1:
                return self._handleError(ErrorCode.InvalidDataHeight)

            width = len(phaseValues[0])

            if width < 1:
                return self._handleError(ErrorCode.InvalidDataWidth)

            type = phaseValues._type_._type_

            if type is ctypes.c_float:
                error = self._library.heds_load_phasevalues_single(handleptr, width, height, ctypes.cast(phaseValues, ctypes.POINTER(ctypes.c_float)), 0, phaseWrap, loadFlags)
                return (error, handle)

            if type is ctypes.c_double:
                error = self._library.heds_load_phasevalues_double(handleptr, width, height, ctypes.cast(phaseValues, ctypes.POINTER(ctypes.c_double)), 0, phaseWrap, loadFlags)
                return (error, handle)

        return self._handleError(ErrorCode.UnsupportedDataFormat)

    ## Loads data from a file.
    # \param imageFilePath The path to an image file you want to load.
    # <br>Supported image file formats are: bmp, cur, dds, gif, icns, ico, jpeg, jpg, pbm, pgm, png, ppm, svg, svgz, tga, tif, tiff, wbmp, webp, xbm, xpm.
    # <br>For holographic data, we recommend not to use any lossy compressed formats, like jpg. Instead please use uncompressed formats (e.g. bmp) or lossless compressed formats (e.g. png).
    # \param loadFlags The flags used when loading the data.
    # \return A tuple (errorCode, handle). You can check if errorCode is ErrorCode.NoError.
    def loadDataFromFile(self, imageFilePath, loadFlags = LoadFlags.Default):
        abspath = os.path.abspath(imageFilePath)

        isUnicode = (isinstance(abspath, str)) if isPython3 else (isinstance(abspath, unicode))

        handle = Datahandle(self)
        handleptr = ctypes.pointer(handle)

        if isUnicode:
            if not os.path.exists(abspath):
                return self._handleError(ErrorCode.FileNotFound)

            error = self._library.heds_load_data_fromfile_unicode(handleptr, abspath, loadFlags)
        else:
            # we do not check if the file exists here because this would fail for utf-8 strings
            error = self._library.heds_load_data_fromfile_utf8(handleptr, abspath, loadFlags)

        return (error, handle)

    ## Updates a handle with the latest information.
    # \param handle The handle we want to update.
    # \return ErrorCode.NoError when there is no error.
    def updateDatahandle(self, handle):
        if not isinstance(handle, Datahandle) or handle.id < 1:
            return ErrorCode.InvalidHandle

        handleptr = ctypes.pointer(handle)

        return self._library.heds_datahandle_update(handleptr)

    ## Releases the data associated with a datahandle id. This function is not required when using the \ref Datahandle class instead of the ids.
    # \param datahandleid The id of the handle you want to release
    # \return No return value.
    def releaseDatahandle(self, datahandleid):
        if datahandleid > 0:
            self._library.heds_datahandle_release_id(datahandleid)

    ## Releases all data associated with handles. This function is not required when using the \ref Datahandle class instead of the ids.
    # \return No return value.
    def releaseAllDatahandles(self):
        self._library.heds_datahandle_release_all()

    ## Updates a handle with the latest information and applies/writes the flagged values.
    # \param handle The handle whose values we want to apply.
    # \param appliedValues Specifies the values which will be applied.
    # \return ErrorCode.NoError when there is no error.
    def datahandleApplyValues(self, handle, appliedValues):
        if not isinstance(handle, Datahandle) or handle.id < 1:
            return ErrorCode.InvalidHandle

        handleptr = ctypes.pointer(handle)

        return self._library.heds_datahandle_apply(handleptr, appliedValues)

    ## Waits until a data handle has reached a certain state.
    # \param handle The handle we want to wait for.
    # \param state The state to wait for.
    # \param timeOutInMs The time in milliseconds to wait before returning with the error ErrorCode.WaitForHandleTimedOut.
    # \return ErrorCode.NoError when there is no error.
    def datahandleWaitFor(self, handle, state, timeOutInMs = 4000):
        if SLMInstance._isID(handle):
            # create DataHandle for this id
            h = Datahandle(self)
            h.id = handle
            handle = h

        if isinstance(handle, Datahandle):
            handleptr = ctypes.pointer(handle)

            return self._library.heds_datahandle_waitfor(handleptr, state, timeOutInMs)

        return ErrorCode.InvalidHandle

    ## Sets Zernike parameter values for the SLM.
    # \param values The Zernike values to set. Refer to \ref holoeye.slmdisplaysdk.heds_types.ZernikeValues for details. Pass zero to disable Zernike again.
    # \return ErrorCode.NoError when there is no error.
    def zernike(self, values):
        valuesCount = 0 if values is None else len(values)
        
        # handle resetting the values
        if valuesCount < 1:
            return self._library.heds_slm_zernike(ctypes.c_void_p(0))
        
        if valuesCount > ZernikeValues.COUNT:
            return self._handleError(ErrorCode.InvalidDataWidth)
        
        # try to forward the data without any conversion
        if supportNumPy and isinstance(values, numpy.ndarray):
            type = values.dtype.type

            if type is numpy.single:
                return self._library.heds_slm_zernike(ctypes.cast(values.ctypes, ctypes.POINTER(ctypes.c_float)), valuesCount)

        elif isinstance(values, ctypes.Array):
            type = values._type_._type_

            if type is ctypes.c_float:
                return self._library.heds_slm_zernike(ctypes.cast(values, ctypes.POINTER(ctypes.c_float)), valuesCount)
            
        # try to convert the data
        floats = (ctypes.c_float * valuesCount)()
        
        for i in range(valuesCount):
            floats[i] = float(values[i])

        return self._library.heds_slm_zernike(ctypes.cast(floats, ctypes.POINTER(ctypes.c_float)), valuesCount)

    ## Loads Zernike parameters from a HOLOEYE Zernike parameter file, for example saved by HOLOEYE SLM Pattern Generator.
    ## The loaded Zernike parameters are directly applied to the SLM.
    # \param filename The Zernike parameter filename to load.
    # \return ErrorCode.NoError when there is no error.
    def zernikeLoadFromFile(self, filename):
        abspath = os.path.abspath(filename)

        isUnicode = (isinstance(abspath, str)) if isPython3 else (isinstance(abspath, unicode))

        if isUnicode:
            if not os.path.exists(abspath):
                return ErrorCode.FileNotFound

            error = self._library.heds_slm_zernike_load_unicode(abspath)
        else:
            # we do not check if the file exists here because this would fail for utf-8 strings
            error = self._library.heds_slm_zernike_load_utf8(abspath)

        return ErrorCode.NoError

    ## Loads and sets a wavefront compensation.
    # \param filename The wavefront compensation H5 file to load.
    # \param wavelength_nm The wavelength in nanometers to load the wavefront compensation for.
    # \param flags Modify how the wavefront compensation will be shown.
    # \param shift_x Shift the wavefront compensation field in x direction.
    # \param shift_y Shift the wavefront compensation field in y direction.
    # \return ErrorCode.NoError when there is no error.
    def wavefrontcompensationLoad(self, filename, wavelength_nm, flags = WavefrontcompensationFlags.NoFlag, shift_x = 0, shift_y = 0):
        abspath = os.path.abspath(filename)

        isUnicode = (isinstance(abspath, str)) if isPython3 else (isinstance(abspath, unicode))

        if isUnicode:
            if not os.path.exists(abspath):
                return ErrorCode.FileNotFound

            error = self._library.heds_slm_wavefrontcompensation_load_unicode(abspath, wavelength_nm, flags, shift_x, shift_y)
        else:
            # we do not check if the file exists here because this would fail for utf-8 strings
            error = self._library.heds_slm_wavefrontcompensation_load_utf8(abspath, wavelength_nm, flags, shift_x, shift_y)

        return error

    ## Clears any loaded wavefront compensation.
    # \return ErrorCode.NoError when there is no error.
    def wavefrontcompensationClear(self):
        return self._library.heds_slm_wavefrontcompensation_clear()
