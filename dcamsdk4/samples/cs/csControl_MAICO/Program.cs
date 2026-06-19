/**
 @file control_maico.cpp
 @date 2025-03-27

 @copyright Copyright (C) 2018-2025 Hamamatsu Photonics K.K.. All rights reserved.

 @brief		Sample code to control MAICO(C15890/C17290).
 @details	This program controls MAICO with subunit properties.
 @details	This program does not work with all cameras.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using Hamamatsu.DCAM4;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace csControl_MAICO
{
    class Program
    {
        /**
         * @brief Retrieves properties of the subunits.
         * @param myDcam MyDcam instance.
         * @return True if successful, otherwise false.
         */
        static bool GetSubunitProperties( MyDcam myDcam )
        {
            bool ret = true;
            // Get maximum number of subunit
            MyDcamProp prop = new MyDcamProp(myDcam, DCAMIDPROP.NUMBEROF_SUBUNIT);
            double value = 0;
            bool lasterr = prop.getvalue(ref value);
            if( !lasterr )
                return false;
            Int32 subunitNum = (Int32)value;
            UInt32 i = 0;
            Console.WriteLine("=========================================================================");
            Console.WriteLine("[#] Subunit Info\t\tControl\t\tLaserPower\tPMTGain");
            Console.WriteLine("-------------------------------------------------------------------------");
            for (i = 0; i < subunitNum; i++)
            {
                // check whether is the subunit installed
                string control = string.Empty;
                prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_CONTROL.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * i);
                lasterr = prop.getvalue(ref value);

                if (!lasterr)
                {
                    Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:SUBUNIT_CONTROL" );
                    return false;
                }
                if (value == DCAMPROP.SUBUNIT_CONTROL.NOTINSTALLED)
                {
                    Console.WriteLine($"[{i}] Not Installed\t\t-\t\t-\t\t-");
                }
                else
                {
                    string strValue = myDcam.dev_getstring(new DCAMIDSTR(DCAMIDSTR.SUBUNIT_INFO1 + (UInt32)i));
                    if (value == DCAMPROP.SUBUNIT_CONTROL.OFF)
                        control = "OFF";
                    else if (value == DCAMPROP.SUBUNIT_CONTROL.ON)
                        control = "ON";
                    // get default values
                    UInt32 power = 0;
                    double gain = 0.0;
                    prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_LASERPOWER.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * i);
                    lasterr = prop.getvalue(ref value);
                    if (!lasterr)
                    {
                        Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:SUBUNIT_LASERPOWER");
                        ret = false;
                        break;
                    }
                    else
                    {
                        power = (UInt32)value;
                        prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_PMTGAIN.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * i);
                        lasterr = prop.getvalue(ref value);
                        if (!lasterr)
                        {
                            Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ rop_getvalue(), IDPROP:SUBUNIT_PMTGAIN");
                            ret = false;
                            break;
                        }
                        else
                        {
                            gain = value;
                            Console.WriteLine($"[{i}] {strValue}\t{control}\t\t{power}\t\t{gain}");
                        }
                    }

                }
            }

            return ret;
        }

        /**
         * @brief Sets properties of each subunits.
         * @param myDcam MyDcam instance.
         * @return True if successful, otherwise false.
         */
        static bool SetSubunitProperties(MyDcam myDcam)
        {
            bool ret = true;
            // Get maximum number of subunit
            MyDcamProp prop = new MyDcamProp(myDcam, DCAMIDPROP.NUMBEROF_SUBUNIT);
            double value = 0;
            bool lasterr = prop.getvalue(ref value);
            if (!lasterr)
            {
                return false;
            }
            Int32 subunitNum = (Int32)value;
            UInt32 i = 0;
            // Turn on to installed subunit
            Console.WriteLine("\nSet SUBUNIT_CONTROL to \"ON\" for all installed subunits");
            for (i = 0; i < subunitNum; i++)
            {
                // check whether is the subunit installed
                prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_CONTROL.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * i);
                lasterr = prop.getvalue(ref value);

                if (value != DCAMPROP.SUBUNIT_CONTROL.NOTINSTALLED)
                {
                    lasterr = prop.setvalue(DCAMPROP.SUBUNIT_CONTROL.ON);

                    if (!lasterr)
                    {
                        Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_setvalue(), IDPROP:SUBUNIT_CONTROL" );
                        return false;
                    }
                    /*
                     // Set Laser Power and PMT Gain to the appropriate value.
                    prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_LASERPOWER.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * i); 
                    lasterr = prop.setvalue(30);

                    if (!lasterr)
                    {
                        Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ rop_setvalue(), IDPROP:SUBUNIT_LASERPOWER" );
                        ret = false;
                        break;
                    }
                    else
                    {
                        prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_PMTGAIN.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * i);
                        lasterr = prop.setvalue(0.7);
                        if (!lasterr)
                        {
                            Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_setvalue(), IDPROP:SUBUNIT_PMTGAIN" );
                            ret = false;
                            break;
                        }
                    }
                    */
                }
            }

            return ret;
        }


        /**
         * @brief Copy image to the specified buffer by the specified area
         * @param myDcam    MyDcam instance.
         * @param iFrame    Frame index to copy.
         * @param buf       Buffer to store the copied image data.
         * @param Rowbytes  Image Rowbytes.
         * @param Width     Image Width.
         * @param Height    Image Height.
         * @return True if successful, otherwise false.
         */
        static bool CopyImage( MyDcam myDcam, Int32 iFrame, byte[] buf, Int32 Rowbytes, Int32 Width, Int32 Height)
        {
            bool ret = true;
            bool err = true;
            GCHandle handle = GCHandle.Alloc(buf, GCHandleType.Pinned);
            // Prepare frame param
            DCAMBUF_FRAME bufframe = new DCAMBUF_FRAME
            {
                size = Marshal.SizeOf(typeof(DCAMBUF_FRAME)),
                iFrame = iFrame,
                buf = handle.AddrOfPinnedObject(),
                rowbytes = Rowbytes,
                left = 0,
                top = 0,
                width = Width,
                height = Height
            };
            // Access image
            err = myDcam.buf_copyframe(ref bufframe);
            if (!err)
            {
                ret = false;
                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ buf_copyframe()");
            }

            return ret;
        }

        /**
         * @brief Get image information from properties.
         * @param myDcam    MyDcam instance.
         * @param PixelType DCAM_PIXELTYPE value.
         * @param Width     Image width.
         * @param Rowbytes  Image rowbytes.
         * @param Height    Image height.
         */
        static void GetImageInformation(MyDcam myDcam, ref Int32 PixelType, ref Int32 Width, ref Int32 Rowbytes, ref Int32 Height)
        {
            bool err;
            double v = 0.0;
            MyDcamProp prop = new MyDcamProp(myDcam, DCAMIDPROP.IMAGE_PIXELTYPE);

            // Image pixel type (DCAMPIXELTYPE_MONO16, MONO8, ...)
            err = prop.getvalue(ref v);
            if (!err)
            {
                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:IMAGE_PIXELTYPE");
                return;
            }
            else
            {
                PixelType = (Int32)v;
            }

            // Image width
            prop.m_idProp = DCAMIDPROP.IMAGE_WIDTH;
            err = prop.getvalue(ref v);
            if (!err)
            {
                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:IMAGE_WIDTH");
                return;
            }
            else
            {
                Width = (Int32)v;
            }

            // Image row bytes
            prop.m_idProp = DCAMIDPROP.IMAGE_ROWBYTES;
            err = prop.getvalue(ref v);
            if (!err)
            {
                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:IMAGE_ROWBYTES");
                return;
            }
            else
            {
                Rowbytes = (int)v;
            }

            // Image height
            prop.m_idProp = DCAMIDPROP.IMAGE_HEIGHT;
            err = prop.getvalue(ref v);
            if (!err)
            {
                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:IMAGE_HEIGHT");
                return;
            }
            else
            {
                Height = (Int32)v;
            }
        }

        /**
         * @brief Sample used to process image after capturing.
         * @param myDcam    MyDcam instance.
         * @param nSubunit	Maximum number of installable subunits.
         */
        static void SampleAccessImage(MyDcam myDcam, UInt32 nSubunit)
        {
            bool err = true;

            // Transfer info param
            DCAMCAP_TRANSFERINFO captransferinfo = new DCAMCAP_TRANSFERINFO
            {
                size = Marshal.SizeOf(typeof(DCAMCAP_TRANSFERINFO))
            };

            // Get number of captured images
            err = myDcam.cap_transferinfo(ref captransferinfo.nNewestFrameIndex, ref captransferinfo.nFrameCount);
            if (!err)
            {
                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ cap_transferinfo()");
                return;
            }

            if (captransferinfo.nFrameCount < 1)
            {
                Console.WriteLine("No captured image");
                return;
            }

            // Get image information
            Int32 pixeltype = 0, width = 0, rowbytes = 0, height = 0;
            GetImageInformation(myDcam, ref pixeltype, ref width, ref rowbytes, ref height);

            if (pixeltype != DCAM_PIXELTYPE.MONO16)
            {
                Console.WriteLine("Not implemented");
                return;
            }

            Int32 bufsize = rowbytes * height;
            byte[] buf = new byte[bufsize];

            int iFrame = captransferinfo.nNewestFrameIndex; // Latest frame
            // Copy whole image
            CopyImage(myDcam, iFrame, buf, rowbytes, width, height);

            // Get subunit image width
            MyDcamProp prop = new MyDcamProp(myDcam, DCAMIDPROP.SUBUNIT_IMAGEWIDTH);
            double value = 0.0;
            err = prop.getvalue(ref value);
            int subunitWidth = 0;
            if (!err)
            {
                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:SUBUNIT_IMAGEWIDTH");
                return;
            }
            else
            {
                subunitWidth = (int)value;
            }

            // Process each subunit image
            int subunitRowBytes = subunitWidth * 2;
            int subunitBufSize = subunitRowBytes * height;
            byte[] subunitBuffer = new byte[subunitBufSize];

            for (UInt32 iSubunit = 0; iSubunit < nSubunit; iSubunit++)
            {
                Array.Clear(subunitBuffer, 0, subunitBufSize);
                prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_CONTROL.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * iSubunit);
                err = prop.getvalue(ref value);
                if (!err)
                {
                    Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:SUBUNIT_CONTROL");
                    break;
                }

                if (value == DCAMPROP.SUBUNIT_CONTROL.OFF)
                {
                    Console.WriteLine($"Subunit Top Offset Bytes[{iSubunit}] : OFF");
                }
                else if (value == DCAMPROP.SUBUNIT_CONTROL.NOTINSTALLED)
                {
                    Console.WriteLine($"Subunit Top Offset Bytes[{iSubunit}] : NOT INSTALLED");
                }
                else
                {
                    prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_TOPOFFSETBYTES.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * iSubunit);
                    err = prop.getvalue(ref value);

                    if (!err)
                    {
                        Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:SUBUNIT_TOPOFFSETBYTES");
                        break;
                    }
                    else
                    {
                        Console.WriteLine($"Subunit Top Offset Bytes[{iSubunit}] : {(int)value}");
                    }

                    int srcOffset = (int)value;
                    for (int i = 0; i < height; i++)
                    {
                        Buffer.BlockCopy(buf, srcOffset, subunitBuffer, i * subunitRowBytes, subunitRowBytes);
                        srcOffset += rowbytes;
                    }

                    prop.m_idProp = new DCAMIDPROP((UInt32)DCAMIDPROP.SUBUNIT_WAVELENGTH.getidprop() + (UInt32)DCAMIDPROP._SUBUNIT.getidprop() * iSubunit);
                    err = prop.getvalue(ref value);

                    if (!err)
                    {
                        Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP:SUBUNIT_WAVELENGTH");
                    }
                    else
                    {
                        string filename;
                        filename = $"subunit_{(int)value}_nm.raw";
                        OutputData(filename, subunitBuffer, subunitBufSize);
                    }
                }
            }

        }

        /**
         * @brief Writes data to a file.
         * @param filename          Name of the file to write to.
         * @param subunitBuffer	    Buffer containing the data to write.
         * @param subunitBufSize	Size of the buffer.
         */
        private static void OutputData(string filename, byte[] subunitBuffer, int subunitBufSize)
        {
            using (FileStream fs = new FileStream(filename, FileMode.Create, FileAccess.Write))
            {
                fs.Write(subunitBuffer, 0, subunitBufSize);
            }
        } 

        static void Main(string[] args)
        {
            Console.WriteLine("PRGRAM START");
            DCAMAPI_INIT param = new DCAMAPI_INIT(0);

            DCAMERR lasterr = dcamapi.init(ref param);
            if (lasterr.failed())
            {
                Console.WriteLine($"FAILED: 0x{(int)lasterr:X8} DCAMERR_{lasterr.text()} @ dcamapi_init()");
            }
            else
            {
                MyDcam myDcam = new MyDcam();
                if (myDcam.dev_open(0))
                {
                    string strModel = myDcam.dev_getstring(DCAMIDSTR.MODEL);
                    string strVer = myDcam.dev_getstring(DCAMIDSTR.CAMERAVERSION);

                    if (strModel.StartsWith("C15890") || strModel.StartsWith("C17290"))
                    {
                        Console.WriteLine("Open " + strModel + " ver " + strVer);

                        if (GetSubunitProperties(myDcam)) // If failed, it means HW error is happened.
                        {
                            MyDcamProp prop = new MyDcamProp(myDcam, DCAMIDPROP.NUMBEROF_SUBUNIT);
                            double value = 0;
                            if (!prop.getvalue(ref value))
                            {
                                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ prop_getvalue(), IDPROP : NUMBEROF_SUBUNIT");
                            }
                            else
                            {
                                UInt32 nSubunit = (UInt32)value;

                                if (SetSubunitProperties(myDcam))  // If failed, it means HW error is happened.
                                {
                                    MyDcamWait wait = new MyDcamWait(ref myDcam);
                                    if (!myDcam.buf_alloc(3))
                                    {
                                        Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ buf_alloc()");
                                    }
                                    else
                                    {
                                        if (!myDcam.cap_start())
                                        {
                                            Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ cap_start()");
                                        }
                                        else
                                        {
                                            Console.WriteLine("Start Capture");
                                            DCAMWAIT eventmask = DCAMWAIT.CAPEVENT.FRAMEREADY | DCAMWAIT.CAPEVENT.STOPPED;
                                            DCAMWAIT eventhappened = DCAMWAIT.NONE;
                                            if (!wait.start(eventmask, ref eventhappened))
                                            {
                                                Console.WriteLine($"FAILED: 0x{(int)myDcam.m_lasterr:X8} DCAMERR_{myDcam.m_lasterr.text()} @ wait_start()");
                                            }
                                            else
                                            {
                                                myDcam.cap_stop();
                                                Console.WriteLine("Stop Capture");
                                                Console.WriteLine("Access Image");
                                                SampleAccessImage(myDcam, nSubunit);
                                            }
                                            myDcam.buf_release();

                                        }
                                    }
                                }
                            }

                        }
                        myDcam.dev_close();
                    }
                    else
                        Console.WriteLine("This program is only for C15890 or C17290!");
                }
            } 
            MyDcamApi.uninit();
            Console.WriteLine("PRGRAM END");
            return;
        }

    }
}
