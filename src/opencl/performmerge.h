#ifndef PERFORMMERGE_H
#define PERFORMMERGE_H

#define CL_TARGET_OPENCL_VERSION 110 // Target OpenCL v1.1
#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include <QObject>
#include <QThread>
#include <QTimer>
#include <array>
#include "streamcommon.h"

class sACNSource;

namespace openCL_experimental {
    class clMergeWorker;

static const unsigned int maxSources = 5; // TODO Don't used fixed size

    class clMerge : public QObject {
        Q_OBJECT

    public:
        clMerge(QObject *parent  = Q_NULLPTR);
        ~clMerge();

        static bool isRunning();
        static const clMergeWorker *getWorker() {
            return workerThread;
        }
        static unsigned int getMergesPerSec();

        typedef std::array<uchar, DMX_SLOT_MAX> slotData_t;
        static void setSourceLevels(
                uint16_t universe, CID cid,
                slotData_t::const_iterator sourceBegin,
                slotData_t::const_iterator sourceEnd);
        static void setSourcePriorities(
                uint16_t universe, CID cid,
                slotData_t::const_iterator sourceBegin,
                slotData_t::const_iterator sourceEnd);
        static CID winningSource(uint16_t universe, uint16_t slot);

    private:
        static clMergeWorker *workerThread;
    };


    class clMergeWorker : public QThread
    {
        Q_OBJECT

        friend class clMerge;

    protected:
        clMergeWorker(QObject *parent = Q_NULLPTR) : QThread(parent) {}
        void run() override;
        void quit() { exit(0); }
        void exit(int retcode = 0);

    signals:        
        void dataChanged(uint16_t universe, clMerge::slotData_t *levels);
        void threadAborted();

    private:
        bool exitThread = false;
        cl_int init();
        void msgErr(QString caller, cl_int err, QString details = QString());

        cl::Context context;
        cl::Kernel kernel;

        cl_int setupBuffers();

        static CID ulong2CID(ulong hi, ulong lo);
        static std::pair<ulong /*hi*/, ulong /*lo*/> CID2Ulong(CID &cid);
        static int universe2Idx(uint16_t universe);
        static int source2Idx(uint16_t universe, CID cid);
        static uint16_t idx2Universe(size_t idx);
        static const sACNSource *idx2Source(uint16_t universe, size_t idx);

        struct bufferData : public QObject {
             cl::Buffer &buffer() {
                 return *m_buffer;
             }

        protected:
             bufferData(cl::Context &context, cl_mem_flags flags, QObject *parent) :
                 QObject(parent), m_context(context), m_flags(flags) {}

            cl_int createBuffer(size_t size, void* ptr) {
                cl_int status = CL_SUCCESS;
                m_buffer.reset(new cl::Buffer(
                                   m_context, m_flags,
                                   size, ptr, &status));
                return status;
            }

        private:
            cl::Context &m_context;
            cl_mem_flags m_flags;
            std::shared_ptr<cl::Buffer> m_buffer;

        };

        template<typename T>
        struct bufferUniverseData : public bufferData {
            static_assert(std::is_same<T, bool>::value == false, "Bool is not allowed, use uint8_t");

            bufferUniverseData(cl::Context &context, cl_mem_flags flags, QObject *parent) :
                bufferData(context, flags, parent)
            {
                setUniverseCount(1);
            }

            cl_int setUniverseCount(uint16_t count) {
                if (getUniverseCount() == count)
                    return CL_SUCCESS;
                m_data.resize(count);
                return createBuffer(size(), data());
            }
            uint16_t getUniverseCount() const {
                return m_data.size();
            }

            T &universe(uint16_t universeIdx) {
                return m_data.at(universeIdx);
            }

            T *universePtr(uint16_t universeIdx) {
                if (universeIdx >= m_data.size())
                    return Q_NULLPTR;
                return &universe(universeIdx);
            }

            size_t size() const {
                return m_data.size() * sizeof(T);
            }

            T *data() {
                return m_data.data();
            }

            void fill(T value) {
                std::fill(m_data.begin(), m_data.end(), value);
            }

        private:
            std::vector<T> m_data;
        };

        template<typename T>
        struct bufferSourceData : public bufferData {
            static_assert(std::is_same<T, bool>::value == false, "Bool is not allowed, use uint8_t");

            bufferSourceData(cl::Context &context, cl_mem_flags flags, QObject *parent) :
                bufferData(context, flags, parent),
                sourceCount(1)
            {
                setUniverseCount(1);
            }

            // TODO allow each universe to have difference source size
            cl_int setUniverseCount(uint16_t count) {
                if (getUniverseCount() == count)
                    return CL_SUCCESS;
                universeCount = count;
                return resize();
            }
            uint16_t getUniverseCount() const {
                return universeCount;
            }

            cl_int setSourceCount(uint16_t count)  {
                if (getSourceCount() == count)
                    return CL_SUCCESS;
                sourceCount = count;
                return resize();
            }
            uint16_t getSourceCount() const {
                return sourceCount;
            }

            T &universe(uint16_t universeIdx, uint16_t sourceIdx) {
                return m_data.at(universeIdx * sourceCount + sourceIdx);
            }

            T *universePtr(uint16_t universeIdx, uint16_t sourceIdx) {
                if (universeIdx >= universeCount || sourceIdx >= sourceCount)
                    return Q_NULLPTR;
                return &universe(universeIdx, sourceIdx);
            }

            size_t size() const {
                return m_data.size() * sizeof(T);
            }

            T *data() {
                return m_data.data();
            }

            void fill(T value) {
                std::fill(m_data.begin(), m_data.end(), value);
            }

        private:
            cl_int resize() {
                m_data.resize(universeCount * sourceCount);
                return createBuffer(size(), data());
            }
            unsigned int universeCount = 0;
            unsigned int sourceCount = 0;
            std::vector<T> m_data;
        };

        bufferUniverseData<clMerge::slotData_t> *mergedLevels = Q_NULLPTR;
        bufferUniverseData<std::array<uint64_t, DMX_SLOT_MAX>> *mergedLevelsSourceCIDHi = Q_NULLPTR;
        bufferUniverseData<std::array<uint64_t, DMX_SLOT_MAX>> *mergedLevelsSourceCIDLo = Q_NULLPTR;
        QElapsedTimer changedUniversesForcedTimer;
        bufferUniverseData<uint8_t> *changedUniverses = Q_NULLPTR;
        bufferUniverseData<uint8_t> *sourceCounts = Q_NULLPTR;
        bufferSourceData<clMerge::slotData_t> *sourceLevels = Q_NULLPTR;
        bufferSourceData<clMerge::slotData_t> *sourcePriorities = Q_NULLPTR;
        bufferSourceData<uint64_t> *sourceCIDsHi = Q_NULLPTR;
        bufferSourceData<uint64_t> *sourceCIDsLo = Q_NULLPTR;

        struct mergesPerSec_s : public QObject {
            public:
                mergesPerSec_s() {
                    timer.setInterval(std::chrono::seconds(1));
                    connect(&timer, &QTimer::timeout,
                            this, [=]()
                    {
                        value = counter;
                        counter = 0;
                    });
                    timer.start();
                }
                void countMerge() { counter++; }
                operator unsigned int() const { return value; }
            private:
                unsigned int value = 0;
                int counter = 0;
                QTimer timer;
        } mergesPerSec;
    };
}; // namespace

#endif // PERFORMMERGE_H
