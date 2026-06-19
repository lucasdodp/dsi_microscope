/**
 @file control_maico.cpp
 @date 2025-03-27

 @copyright Copyright (C) 2018-2025 Hamamatsu Photonics K.K.. All rights reserved.

 @brief		Sample code to control MAICO(C15890/C17290).
 @details	This program controls MAICO with subunit properties.
 @details	This program does not work with all cameras.
 @remarks	dcamprop_setvalue
 @remarks	dcambuf_copyframe
 */

#include "../misc/console4.h"
#include "../misc/common.h"

/**
 @brief Retrieves properties of the subunits.
 @param hdcam DCAM handle
 @return result of getting subunits properties
*/
BOOL get_subunitproperties(HDCAM hdcam)
{
	DCAMERR err = DCAMERR_SUCCESS;
	BOOL ret = TRUE;

	//get maximum number of subunit
	double value = 0.0;
	err = dcamprop_getvalue(hdcam, DCAM_IDPROP_NUMBEROF_SUBUNIT, &value);
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:NUMBEROFSUBUNIT");
		return FALSE;
	}

	int32 subunitNum = (int32)value;
	char subunitInfo[256];
	DCAMDEV_STRING	param;
	memset(&param, 0, sizeof(param));
	param.size = sizeof(param);
	param.text = subunitInfo;
	param.textbytes = 256;
	int32 i = 0;

	int32 number = 0;
	printf("=========================================================================\n");
	printf("[#] Subunit Info\t\tControl\t\tLaserPower\tPMTGain\n");
	printf("-------------------------------------------------------------------------\n");


	for (i = 0; i < subunitNum; i++)
	{
		// check whether is the subunit installed
		int32 offset = DCAM_IDPROP__SUBUNIT * i;
		char control[32];
		err = dcamprop_getvalue(hdcam, DCAM_IDPROP_SUBUNIT_CONTROL + offset, &value);
		if (failed(err))
		{
			dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:SUBUNIT_CONTROL");
			ret = FALSE;
			break;
		}
		else
		{
			if (value == DCAMPROP_SUBUNIT_CONTROL__NOTINSTALLED)
				printf("[%d] Not Installed\t\t-\t\t-\t\t-\n", i);

			else
			{
				param.iString = DCAM_IDSTR_SUBUNIT_INFO1 + i;
				err = dcamdev_getstring(hdcam, &param);

				if (value == DCAMPROP_SUBUNIT_CONTROL__OFF)
					strcpy_s(control, 32, "OFF");
				else if (value == DCAMPROP_SUBUNIT_CONTROL__ON)
					strcpy_s(control, 32, "ON");

				// get default values
				int32 power = 0;
				double gain = 0.0;
				err = dcamprop_getvalue(hdcam, DCAM_IDPROP_SUBUNIT_LASERPOWER + offset, &value);
				if (failed(err))
				{
					dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:SUBUNIT_LASERPOWER");
					ret = FALSE;
					break;
				}
				else
				{
					power = (int32)value;
					err = dcamprop_getvalue(hdcam, DCAM_IDPROP_SUBUNIT_PMTGAIN + offset, &gain);
					if (failed(err))
					{
						dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:SUBUNIT_PMTGAIN");
						ret = FALSE;
						break;
					}
					else
					{
						printf("[%d] %s\t%s\t\t%d\t\t%lf\n", i, subunitInfo, control, power, gain);
					}
				}
			}
		}
	}

	return ret;
}

/**
 @brief sets properties of each subunits
 @param hdcam DCAM handle
 @return result of setting subunits properties
*/
BOOL set_subunitproperties(HDCAM hdcam)
{
	DCAMERR err = DCAMERR_SUCCESS;
	int32 ret = 0;

	//get maximum number of subunit
	double value = 0.0;
	err = dcamprop_getvalue(hdcam, DCAM_IDPROP_NUMBEROF_SUBUNIT, &value);
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:NUMBEROFSUBUNIT");
		return FALSE;
	}

	// Turn on to installed subunit
	int32 i = 0;
	int32 nSubunit = (int32)value;
	printf("\nSet SUBUNIT_CONTROL to \"ON\" for all installed subunits\n");
	for (i = 0; i < nSubunit; i++)
	{
		int32 offset = DCAM_IDPROP__SUBUNIT * i;
		err = dcamprop_getvalue(hdcam, DCAM_IDPROP_SUBUNIT_CONTROL + offset, &value);
		if (failed(err))
		{
			dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:SUBUNIT_CONTROL");
			ret = 1;
			break;
		}

		if (value != DCAMPROP_SUBUNIT_CONTROL__NOTINSTALLED)
		{
			err = dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBUNIT_CONTROL + offset, DCAMPROP_SUBUNIT_CONTROL__ON);
			if (failed(err))
			{
				dcamcon_show_dcamerr(hdcam, err, "dcamprop_setvalue()", "IDPROP:SUBUNIT_CONTROL, VALUE: ON");
				ret = 1;
			}
			/*
			// Set Laswer Power and PMT Gain to the appropiate value.
			err = dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBUNIT_LASERPOWER + offset, 30);
			if (failed(err))
			{
				dcamcon_show_dcamerr(hdcam, err, "dcamprop_setvalue()", "IDPROP:SUBUNIT_LASERPOWER");
				ret = 1;
			}

			err = dcamprop_setvalue(hdcam, DCAM_IDPROP_SUBUNIT_PMTGAIN + offset, 0.7);
			if (failed(err))
			{
				dcamcon_show_dcamerr(hdcam, err, "dcamprop_setvalue()", "IDPROP:SUBUNIT_PMTGAIN");
				ret = 1;
			}
			*/
		}
	}

	if (ret > 0) // HW error is occured.
		return FALSE;
	else 
		return TRUE;
}


/**
 @brief	Copy image to the specified buffer by the specified area.
 @param	hdcam		DCAM handle
 @param iFrame		frame index
 @param buf		    buffer to copy image
 @param rowbytes	image rowbytes
 @param width		image width
 @param height		image height
 @return	result of copy image
 */
BOOL copy_targetarea(HDCAM hdcam, int32 iFrame, void* buf, int32 rowbytes, int32 width, int32 height)
{
	DCAMERR err;

	// prepare frame param
	DCAMBUF_FRAME bufframe;
	memset(&bufframe, 0, sizeof(bufframe));
	bufframe.size = sizeof(bufframe);
	bufframe.iFrame = iFrame;

	// set user buffer information and copied ROI
	bufframe.buf = buf;
	bufframe.rowbytes = rowbytes;
	bufframe.left = 0;
	bufframe.top = 0;
	bufframe.width = width;
	bufframe.height = height;


	// access image
	err = dcambuf_copyframe(hdcam, &bufframe);
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcambuf_copyframe()");
		return FALSE;
	}

	return TRUE;
}

/**
 @brief	Get image information from properties.
 @param	hdcam		DCAM handle
 @param pixeltype	DCAM_PIXELTYPE value
 @param width		image width
 @param rowbytes	image rowbytes
 @param height		image height
 */
void get_image_information(HDCAM hdcam, int32& pixeltype, int32& width, int32& rowbytes, int32& height)
{
	DCAMERR err;

	double v;

	// image pixel type(DCAM_PIXELTYPE_MONO16, MONO8, ... )
	err = dcamprop_getvalue(hdcam, DCAM_IDPROP_IMAGE_PIXELTYPE, &v);
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:IMAGE_PIXELTYPE");
		return;
	}
	else
		pixeltype = (int32)v;

	// image width
	err = dcamprop_getvalue(hdcam, DCAM_IDPROP_IMAGE_WIDTH, &v);
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:IMAGE_WIDTH");
		return;
	}
	else
		width = (int32)v;

	// image row bytes
	err = dcamprop_getvalue(hdcam, DCAM_IDPROP_IMAGE_ROWBYTES, &v);
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:IMAGE_ROWBYTES");
		return;
	}
	else
		rowbytes = (int32)v;

	// image height
	err = dcamprop_getvalue(hdcam, DCAM_IDPROP_IMAGE_HEIGHT, &v);
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:IMAGE_HEIGHT");
		return;
	}
	else
		height = (int32)v;
}

/**
 @brief	Sample used to process image after capturing.
 @param	hdcam		DCAM handle
 @param nSubunit	maximum number of installable subunits
 @sa	get_image_information, copy_targetarea
 */
void sample_access_image(HDCAM hdcam, int32 nSubunit)
{
	DCAMERR err;

	// transferinfo param
	DCAMCAP_TRANSFERINFO captransferinfo;
	memset(&captransferinfo, 0, sizeof(captransferinfo));
	captransferinfo.size = sizeof(captransferinfo);

	// get number of captured image
	err = dcamcap_transferinfo(hdcam, &captransferinfo);
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcamcap_transferinfo()");
		return;
	}

	if (captransferinfo.nFrameCount < 1)
	{
		printf("not capture image\n");
		return;
	}

	// get image information
	int32 pixeltype = 0, width = 0, rowbytes = 0, height = 0;
	get_image_information(hdcam, pixeltype, width, rowbytes, height);

	if (pixeltype != DCAM_PIXELTYPE_MONO16)
	{
		printf("not implement\n");
		return;
	}

	int32 bufsize = rowbytes * height;
	char* buf = new char[bufsize];
	memset(buf, 0, bufsize);

	int iFrame = captransferinfo.nNewestFrameIndex; // latest frame
	// copy whole image
	copy_targetarea(hdcam, iFrame, buf, rowbytes, width, height);

	// get subunit image width
	double value = 0.0;
	err = dcamprop_getvalue(hdcam, DCAM_IDPROP_SUBUNIT_IMAGEWIDTH, &value);
	int32 subunitWidth = 0;
	if (failed(err))
	{
		dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:SUBUNIT_IMAGEWIDTH");
		return;
	}
	else
		subunitWidth = (int32)value;


	int32 iSubunit = 0;
	// process each subunit image
	int32 subunitRowBytes = subunitWidth * 2;
	int32 subunitBufSize = subunitRowBytes * height;
	char* subunitBuffer = new char[subunitBufSize];
	int32 i = 0;
	char* pSrc = 0, * pDst = 0;
	for (iSubunit = 0; iSubunit < nSubunit; iSubunit++)
	{
		memset(subunitBuffer, 0, subunitBufSize * sizeof(char));
		int32 offset = DCAM_IDPROP__SUBUNIT * iSubunit;
		double value = 0.0;

		err = dcamprop_getvalue(hdcam, DCAM_IDPROP_SUBUNIT_CONTROL + offset, &value);
		if (failed(err))
		{
			dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:SUBUNIT_CONTROL");
			break;
		}
		if (value == DCAMPROP_SUBUNIT_CONTROL__OFF)
			printf("Subunit Top Offset Bytes[%d] : OFF\n", iSubunit);
		else if (value == DCAMPROP_SUBUNIT_CONTROL__NOTINSTALLED)
			printf("Subunit Top Offset Bytes[%d] : NOT INSTALLED\n", iSubunit);
		else
		{
			err = dcamprop_getvalue(hdcam, DCAM_IDPROP_SUBUNIT_TOPOFFSETBYTES + offset, &value);
			if (failed(err))
			{
				dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:SUBUNIT_TOPOFFSETBYTES");
				break;
			}
			else
				printf("Subunit Top Offset Bytes[%d] : %d\n", iSubunit, (int32)value);

			pSrc = buf + (int32)value;
			pDst = subunitBuffer;
			for (i = 0; i < height; i++)
			{
				memcpy_s(pDst, subunitRowBytes, pSrc, subunitRowBytes);
				pDst += subunitRowBytes;
				pSrc += rowbytes;
			}

			char filename[MAX_PATH];
			err = dcamprop_getvalue(hdcam, DCAM_IDPROP_SUBUNIT_WAVELENGTH + DCAM_IDPROP__SUBUNIT * iSubunit, &value);
			if (failed(err))
				dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPROP:SUBUNT_WAVELENGTH");
			else
			{
				sprintf_s(filename, sizeof(filename), "subunit_%d_nm.raw", (int32)value);
				output_data(filename, subunitBuffer, subunitBufSize);
			}
		}
	}
	delete[] subunitBuffer;
	delete[] buf;

}


int main(int argc, char* const argv[])
{
	printf("PROGRAM START\n");

	int	ret = 0;

	DCAMERR err;

	// initialize DCAM-API and open device
	HDCAM hdcam;
	hdcam = dcamcon_init_open();
	if (hdcam == NULL)
	{
		// failed open DCAM handle
		ret = 1;
	}
	else
	{
		// show device information
		dcamcon_show_dcamdev_info(hdcam);

		char cameraName[256];
		DCAMDEV_STRING	param;
		memset(&param, 0, sizeof(param));
		param.size = sizeof(param);
		param.text = cameraName;
		param.textbytes = 256;
		param.iString = DCAM_IDSTR_MODEL;

		err = dcamdev_getstring(hdcam, &param);
		if (failed(err))
		{
			dcamcon_show_dcamerr(hdcam, err, "dcamdev_getstring(DCAM_IDSTR_MODEL)\n");
			ret = 1;
		}
		else
		{
			if (strncmp("C15890", cameraName, 6) == 0 || strncmp("C17290", cameraName, 6) == 0)
			{
				if (!get_subunitproperties(hdcam))
					ret = 1; // It means HW error is happened.
				else
				{

					// Turn on to installed subunit
					int32 i = 0;
					int32 offset = 0;
					double value = 0.0;
					err = dcamprop_getvalue(hdcam, DCAM_IDPROP_NUMBEROF_SUBUNIT, &value);
					if (failed(err))
					{
						dcamcon_show_dcamerr(hdcam, err, "dcamprop_getvalue()", "IDPORP:NUMBEROF_SUBUNIT");
						ret = 1;
					}
					else
					{
						int32 nSubunit = (int32)value;
						if (!set_subunitproperties(hdcam))
							ret = 1;
						else
						{ 
						
							// open wait handle
							DCAMWAIT_OPEN	waitopen;
							memset(&waitopen, 0, sizeof(waitopen));
							waitopen.size = sizeof(waitopen);
							waitopen.hdcam = hdcam;

							err = dcamwait_open(&waitopen);
							if (failed(err))
							{
								dcamcon_show_dcamerr(hdcam, err, "dcamwait_open()");
								ret = 1;
							}
							else
							{
								HDCAMWAIT hwait = waitopen.hwait;

								int32 number_of_buffer = 3;
								err = dcambuf_alloc(hdcam, number_of_buffer);
								if (failed(err))
								{
									dcamcon_show_dcamerr(hdcam, err, "dcambuf_alloc()");
									ret = 1;
								}
								else
								{
									// start capture
									err = dcamcap_start(hdcam, DCAMCAP_START_SNAP);
									if (failed(err))
									{
										dcamcon_show_dcamerr(hdcam, err, "dcamcap_start()");
										ret = 1;
									}
									else
									{
										printf("\nStart Capture\n");

										// set wait param
										DCAMWAIT_START waitstart;
										memset(&waitstart, 0, sizeof(waitstart));
										waitstart.size = sizeof(waitstart);
										waitstart.eventmask = DCAMWAIT_CAPEVENT_FRAMEREADY;
										waitstart.timeout = 1000;

										// wait image
										err = dcamwait_start(hwait, &waitstart);
										if (failed(err))
										{
											dcamcon_show_dcamerr(hdcam, err, "dcamwait_start()");
											ret = 1;
										}

										// stop capture
										dcamcap_stop(hdcam);
										printf("Stop Capture\n");

										// access image
										printf("Access Image\n");
										sample_access_image(hdcam, nSubunit);
									}

									// release buffer
									dcambuf_release(hdcam);
								}

								// close wait handle
								dcamwait_close(hwait);
							}


						}
					}
				}
			}
			else
			{
				printf("This program only for C15890 or C17290!\n");
				ret = 1;
			}
		}
		// close DCAM handle
		dcamdev_close(hdcam);
	}

	// finalize DCAM-API
	dcamapi_uninit();

	printf( "PROGRAM END\n" );
	return ret;	// 0:Success, Other:Failure
}