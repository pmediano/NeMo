INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/src
)

FUNCTION(PLUGIN PLUGIN_NAME)
	SET(TARGET ${PLUGIN_NAME}_cpu)
	ADD_LIBRARY(${TARGET} SHARED ${PLUGIN_NAME}.cpp)
	SET_TARGET_PROPERTIES(${TARGET} PROPERTIES OUTPUT_NAME ${PLUGIN_NAME})
	SET_TARGET_PROPERTIES(${TARGET} PROPERTIES DEFINE_SYMBOL NEMO_PLUGIN_EXPORTS)
	INSTALL(TARGETS ${TARGET} DESTINATION ${NEMO_SYSTEM_PLUGIN_DIR}/cpu)
	TARGET_LINK_LIBRARIES(${TARGET} nemo_base)
ENDFUNCTION(PLUGIN)

PLUGIN(Input)
PLUGIN(PoissonSource)
PLUGIN(Izhikevich)
PLUGIN(IF_curr_exp)
PLUGIN(Kuramoto)
PLUGIN(IF_lin)

